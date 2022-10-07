import torch
import numpy as np
from tqdm import tqdm
import random
import scipy
from copy import deepcopy

from src.utils import *
from src.dataloader import *
from src.trainer import *
from src.model import *
from src.config import *


def main_cl(params):
    # ===========================================================================
    # Using Fixed Random Seed
    if params.seed:
        random.seed(params.seed)
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed(params.seed)
        torch.backends.cudnn.deterministic = True
    # Initialize Experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)
    logger.info(params.__dict__)
    # Set domain name
    domain_name = os.path.basename(params.data_path[0])
    if domain_name=='':
        # Remove the final char '\' in the path
        domain_name = os.path.basename(params.data_path[0][:-1])
    # Generate Dataloader 
    ner_dataloader = NER_dataloader(data_path=params.data_path,
                                    domain_name=domain_name,
                                    batch_size=params.batch_size, 
                                    entity_list=params.entity_list,
                                    n_samples=params.n_samples,
                                    is_filter_O=params.is_filter_O,
                                    schema=params.schema,
                                    is_load_disjoin_train=params.is_load_disjoin_train)
    label_list = ner_dataloader.label_list
    entity_list = ner_dataloader.entity_list
    num_classes_all = len(ner_dataloader.entity_list)
    pad_token_id = ner_dataloader.auto_tokenizer.pad_token_id
    class_per_entity = len(params.schema)-1
    
    # Initialize the model for the first group of classes
    if params.model_name in ['bert-base-cased','roberta-base','bert-base-chinese']:
        # BERT-based NER Tagger
        model = BertTagger(output_dim=(1+class_per_entity*params.nb_class_fg), params=params)
    else:
        raise Exception('model name %s is invalid'%params.model_name)
    model.cuda()
    trainer = BaseTrainer(params, model, entity_list, label_list)
    trainer.pad_token_id = pad_token_id

    # ===========================================================================
    # Start training
    total_iter = int((num_classes_all-params.nb_class_fg)/params.nb_class_pg)+1
    assert (num_classes_all-params.nb_class_fg)%params.nb_class_pg==0, "Invalid class number!"
    for iteration in range(total_iter):
        logger.info("=========================================================")   
        logger.info("Beggin training the %d-th iter (total %d iters)"%(iteration+1, 
                                                                        total_iter))     
        logger.info("=========================================================")
        
        best_model_ckpt_name = "best_finetune_domain_%s_iteration_%d.pth"%(
                                domain_name, 
                                iteration)
        best_model_ckpt_path = os.path.join(
            params.dump_path, 
            best_model_ckpt_name
        )
        if params.is_load_common_first_model:
            common_first_model_ckpt_name = "best_finetune_domain_%s_iteration_%d_fg_%d.pth"%(
                                    domain_name, 
                                    iteration,
                                    params.nb_class_fg)
            common_first_model_ckpth_path = os.path.join(
                os.path.dirname(os.path.dirname(params.dump_path)),
                common_first_model_ckpt_name
            )

        # Initialize a new model
        if params.is_from_scratch or iteration == 0:
            # Initialize the model for the first group of classes
            if params.model_name in ['bert-base-cased','roberta-base','bert-base-chinese']:
                # BERT-based NER Tagger
                model = BertTagger(output_dim=(1+class_per_entity*(params.nb_class_fg+iteration*params.nb_class_pg)), params=params)
            else:
                raise Exception('model name %s is invalid'%params.model_name)
            trainer.model = model
            trainer.model.cuda()

            trainer.refer_model = None
            hidden_dim = trainer.model.classifier.hidden_dim
            output_dim = trainer.model.classifier.output_dim
            logger.info("hidden_dim=%d, output_dim=%d"%(hidden_dim,output_dim))

        # Update the architecture of the classifier
        elif iteration == 1:
            trainer.refer_model = deepcopy(trainer.model)
            trainer.refer_model.eval()
            # Change model classifier
            hidden_dim = trainer.model.classifier.hidden_dim
            output_dim = trainer.model.classifier.output_dim
            logger.info("hidden_dim=%d, old_output_dim=%d, new_output_dim=%d"%(
                                        hidden_dim,
                                        output_dim,
                                        class_per_entity*params.nb_class_pg))
            new_fc = SplitCosineLinear(hidden_dim, output_dim, class_per_entity*params.nb_class_pg)

            new_fc.fc0.weight.data = trainer.model.classifier.weight.data[:1] # for O class
            new_fc.fc1.weight.data = trainer.model.classifier.weight.data[1:]
            new_fc.sigma.data = trainer.model.classifier.sigma.data

            trainer.model.classifier = new_fc
            trainer.model.cuda()

        else:
            trainer.refer_model = deepcopy(trainer.model)
            trainer.refer_model.eval()
            # Change model classifier
            hidden_dim = trainer.model.classifier.hidden_dim
            output_dim1 = trainer.model.classifier.fc1.output_dim
            output_dim2 = trainer.model.classifier.fc2.output_dim
            logger.info("hidden_dim=%d, old_output_dim=%d, new_output_dim=%d"%(
                                                            hidden_dim,
                                                            1+output_dim1+output_dim2,
                                                            class_per_entity*params.nb_class_pg))                                                
            new_fc = SplitCosineLinear(hidden_dim, 1+output_dim1+output_dim2, class_per_entity*params.nb_class_pg)

            new_fc.fc0.weight.data = trainer.model.classifier.fc0.weight.data # for O classes
            new_fc.fc1.weight.data[:output_dim1] = trainer.model.classifier.fc1.weight.data
            new_fc.fc1.weight.data[output_dim1:] = trainer.model.classifier.fc2.weight.data
            new_fc.sigma.data = trainer.model.classifier.sigma.data

            trainer.model.classifier = new_fc
            trainer.model.cuda()

        # Update entity list and label list
        if iteration==0:
            new_entity_list = ner_dataloader.entity_list[:params.nb_class_fg]
            all_seen_entity_list = ner_dataloader.entity_list[:params.nb_class_fg]
        else:
            new_entity_list = ner_dataloader.entity_list[\
                                params.nb_class_fg+(iteration-1)*params.nb_class_pg
                                :params.nb_class_fg+iteration*params.nb_class_pg]
            all_seen_entity_list = ner_dataloader.entity_list[\
                                :params.nb_class_fg+iteration*params.nb_class_pg]
        num_classes_new = 1+class_per_entity*len(all_seen_entity_list)
        if iteration>0:
            num_classes_old = num_classes_new - class_per_entity*len(new_entity_list)
        else:
            num_classes_old = 0
        new_classes_list = list(range(num_classes_old,num_classes_new))
        logger.info("All seen entity types = %s"%str(all_seen_entity_list))
        logger.info("New entity types = %s"%str(new_entity_list))
        
        # Prepare data
        dataloader_train, dataloader_dev = ner_dataloader.get_dataloader(
                                                            first_N_classes=-1,
                                                            select_entity_list=new_entity_list,
                                                            phase=['train','dev'],
                                                            is_filter_O=params.is_filter_O,
                                                            reserved_ratio=params.reserved_ratio)
        # for debug
        dataloader_dev_cumul, dataloader_test_cumul = ner_dataloader.get_dataloader(
                                                            first_N_classes=len(all_seen_entity_list),
                                                            select_entity_list=[],
                                                            phase=['dev','test'],
                                                            is_filter_O=False)
        # for debug and comparision 
        if iteration>0 and (params.sample_strategy=='ground_truth'or params.is_MTL):
            if params.is_load_disjoin_train:
                if params.sample_strategy=='ground_truth':
                    dataloader_train_extra, = ner_dataloader.get_dataloader(
                                                    first_N_classes=-1,
                                                    select_entity_list=new_entity_list,
                                                    phase=['train'],
                                                    is_filter_O=False,
                                                    is_ground_truth_train=True)
                elif params.is_MTL:
                    dataloader_train_extra, = ner_dataloader.get_dataloader(
                                                    first_N_classes=len(all_seen_entity_list),
                                                    select_entity_list=[],
                                                    phase=['train'],
                                                    is_filter_O=False,
                                                    is_ground_truth_train=False)
                    dataloader_train = dataloader_train_extra
            else:
                old_entity_list = list(set(all_seen_entity_list)-set(new_entity_list))
                dataloader_train_extra, = ner_dataloader.get_dataloader(
                                                    first_N_classes=len(all_seen_entity_list),
                                                    select_entity_list=[],
                                                    phase=['train'],
                                                    is_filter_O=params.is_filter_O,
                                                    filter_entity_list=old_entity_list)
                assert len(dataloader_train_extra.dataset.y)==len(dataloader_train.dataset.y)
                if params.is_MTL:
                    dataloader_train = dataloader_train_extra

        # for debug
        # dataloader_test_all = ner_dataloader.get_dataloader(
        #                                                     first_N_classes=-1,
        #                                                     select_entity_list=[],
        #                                                     phase=['test'],
        #                                                     is_filter_O=False)

        if iteration==0:
            # build scheduler and optimizer
            trainer.optimizer = torch.optim.SGD(trainer.model.parameters(),
                                            lr=trainer.lr,
                                            momentum=trainer.mu,
                                            weight_decay=trainer.weight_decay)

            trainer.scheduler = torch.optim.lr_scheduler.MultiStepLR(trainer.optimizer,
                                                                milestones=eval(params.schedule),
                                                                gamma=params.gamma)  

        else:
            # iteration>0
            # Update optimizer and scheduler: Fix the embedding of old classes
            if params.is_fix_trained_classifier:
                # if fix the O classifier
                if params.is_unfix_O_classifier:
                    ignored_params = list(map(id, trainer.model.classifier.fc1.parameters()))
                    base_params = filter(lambda p: id(p) not in ignored_params, \
                                trainer.model.parameters())
                    tg_params =[{'params': base_params, 'lr': float(params.stable_lr),
                                'weight_decay': float(params.weight_decay)}, \
                                {'params': trainer.model.classifier.fc1.parameters(), 'lr': 0., 
                                'weight_decay': 0.}]
                else:
                    ignored_params = list(map(id, trainer.model.classifier.fc1.parameters())) + \
                                    list(map(id, trainer.model.classifier.fc0.parameters()))
                    base_params = filter(lambda p: id(p) not in ignored_params, \
                                trainer.model.parameters())
                    tg_params =[{'params': base_params, 'lr': float(params.stable_lr),
                                'weight_decay': float(params.weight_decay)}, \
                                {'params': trainer.model.classifier.fc0.parameters(), 'lr': 0., 
                                'weight_decay': 0.}, \
                                {'params': trainer.model.classifier.fc1.parameters(), 'lr': 0., 
                                'weight_decay': 0.}]
            else:
                tg_params = [{'params': trainer.model.parameters(), 'lr': float(params.stable_lr), 
                            'weight_decay': float(params.weight_decay)}]
            trainer.optimizer = torch.optim.SGD(tg_params, 
                                                momentum=params.mu)
            # last_epoch_or_step = last_global_step if params.is_train_by_steps \
            #                                     else last_global_epoch
            trainer.scheduler = None

        # Scaling the weights in the new classifier(imprint)
        if iteration>0 and params.is_rescale_new_weight and (not params.is_from_scratch):  
            # (1) compute the average norm of old embdding
            old_embedding_norm = trainer.model.classifier.fc1.weight.data.norm(dim=1, keepdim=True)
            average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).cpu().type(torch.DoubleTensor)
            # (2) compute class centers for each new classes (B-/I-)
            class_center_matrix = compute_class_feature_center(dataloader_dev, 
                                        feature_model=trainer.model.encoder, 
                                        select_class_indexes=new_classes_list, 
                                        is_normalize=True,
                                        is_return_flatten_feat_and_Y=False)
            # (3) rescale the norm for each classes (each row) 
            rescale_weight_matrix = F.normalize(class_center_matrix, p=2, dim=-1) * average_old_embedding_norm
            nan_pos_list = torch.where(torch.isnan(rescale_weight_matrix[:,0]))[0]
            for nan_pos in nan_pos_list:
                assert nan_pos%2==1, "Entity not appear in dataloader!!!"
                # replace the weight of I- with B-
                rescale_weight_matrix[nan_pos] = rescale_weight_matrix[nan_pos-1].clone()
            trainer.model.classifier.fc2.weight.data = rescale_weight_matrix.type(torch.FloatTensor).cuda()

        # Evaluation before training the target model
        # logger.info('Before training evaluation')
        # f1_dev, ma_f1_dev, f1_dev_each_class = trainer.evaluate(dataloader_dev, 
        #                                             each_class=True,
        #                                             entity_order=new_entity_list)
        # logger.info("New data: Dev_f1=%.3f, Dev_ma_f1=%.3f, Dev_f1_each_class=%s" % (
        #      f1_dev, ma_f1_dev, str(f1_dev_each_class)
        # ))
        # f1_dev_cuml, ma_f1_dev_cuml, f1_dev_each_class_cuml = trainer.evaluate(dataloader_dev_cumul, 
        #                                             each_class=True,
        #                                             entity_order=all_seen_entity_list)
        # logger.info("Accumulation: Dev_f1=%.3f, Dev_ma_f1=%.3f, Dev_f1_each_class=%s" % (
        #     f1_dev_cuml, ma_f1_dev_cuml, str(f1_dev_each_class_cuml)
        # ))

        # Init training variables
        if iteration==0 and params.first_training_epochs>0:
            training_epochs = params.first_training_epochs
        else:
            training_epochs = params.training_epochs
        no_improvement_num = 0
        best_f1 = -1
        step = 0
        is_finish = False

        # Reset the training epoch if train by steps
        if params.is_train_by_steps:
            steps_per_epoch = int(len(dataloader_train.dataset)/params.batch_size)
            if iteration==0 and params.first_training_steps>0:
                training_epochs = int(params.first_training_steps/steps_per_epoch)+1
            else:
                training_epochs = int(params.training_steps/steps_per_epoch)+1

        # Check if checkpoint exists and continal training on that checkpoint
        if params.is_load_ckpt_if_exists:
            if iteration==0 and params.is_load_common_first_model and os.path.isfile(common_first_model_ckpth_path):
                logger.info("Skip training %d-th iter checkpoint %s exists"%\
                                (iteration+1, common_first_model_ckpth_path))
                training_epochs = 0
            elif os.path.isfile(best_model_ckpt_path):
                logger.info("Skip training %d-th iter checkpoint %s exists"%\
                                (iteration+1, best_model_ckpt_path))
                training_epochs = 0

        # Compute match samples for DCE
        if training_epochs>0 and iteration>0 and (params.is_DCE or params.is_ODCE):
            # (1) ref feature model is fixed, get old features z0
            # 1.1 Compute the feature (flatten) and collect all data (not flatten)
            (refer_flatten_feat_train, refer_flatten_feat_O_train) \
                = compute_feature_by_dataloader(dataloader=dataloader_train,
                                                feature_model=trainer.refer_model.encoder, 
                                                select_label_groups=[
                                                    new_classes_list,
                                                    [ner_dataloader.O_index],
                                                ],
                                                is_normalize=True)
            trainer.dataloader_train = dataloader_train
            num_sentence_all = len(trainer.dataloader_train.dataset.y)
            
            # 1.2 flatten label list and compute the neighbor for each sample
            if params.is_DCE:
                flatten_label_train, pos_matrix = get_flatten_for_nested_list(
                                                trainer.dataloader_train.dataset.y, 
                                                select_labels=new_classes_list,
                                                is_return_pos_matrix=True,
                                                max_seq_length=params.max_seq_length)
                trainer.pos_matrix = pos_matrix
                num_samples_all = len(flatten_label_train)
                assert refer_flatten_feat_train.shape[0] == num_samples_all, \
                        "refer_flatten_feat_train.shape[0]!=num_samples_all !!!"
                # compute the neighbor for each sample
                match_id = get_match_id(refer_flatten_feat_train, params.top_k)
                # save the space
                del refer_flatten_feat_train

            if params.is_ODCE:
                _, O_pos_matrix = get_flatten_for_nested_list(
                                                trainer.dataloader_train.dataset.y, 
                                                select_labels=[ner_dataloader.O_index],
                                                is_return_pos_matrix=True)

                ground_truth_O_pos_matrix_list = []
                old_class_list = []
                if params.sample_strategy=='ground_truth':
                    old_class_list = list(range(1,num_classes_old))
                    for old_class_id in old_class_list:
                        _, ground_truth_O_pos_matrix = get_flatten_for_nested_list(
                                                    dataloader_train_extra.dataset.y, 
                                                    select_labels=[old_class_id],
                                                    is_return_pos_matrix=True)
                        ground_truth_O_pos_matrix_list.append(ground_truth_O_pos_matrix)
                
                refer_flatten_feat_O_train, O_pos_matrix = trainer.select_O_samples(
                                                refer_flatten_feat_O_train, 
                                                O_pos_matrix,
                                                sample_strategy=params.sample_strategy,
                                                sample_ratio=params.sample_ratio,
                                                ground_truth_O_pos_matrix_list=ground_truth_O_pos_matrix_list,
                                                old_class_list = old_class_list)
                if len(O_pos_matrix)>0:
                    trainer.O_pos_matrix = O_pos_matrix
                    num_O_samples_all = refer_flatten_feat_O_train.shape[0]
                    # compute the neighbor for each sample
                    O_match_id = get_match_id(refer_flatten_feat_O_train, params.top_k)
                    # save the space
                    del refer_flatten_feat_O_train
                else:
                    trainer.O_pos_matrix = []
                    num_O_samples_all = 0
                    O_match_id = []
                
        # Start training the target model
        if trainer.scheduler!=None:
            logger.info("Initial lr is %s"%( str(trainer.scheduler.get_last_lr())))
        for e in range(1, training_epochs+1):
            if is_finish:
                break
            logger.info("============== epoch %d ==============" % e)
            # loss list
            loss_list, distill_list, ce_list = [], [], []
            # average loss
            mean_loss = 0.0
            # training acc
            total_cnt, correct_cnt = 0, 0
            # sample count for DCE
            sample_id, O_sample_id, sentence_id = 0, 0, 0
            # update epoch
            trainer.epoch = e

            for X, y in dataloader_train:
                if is_finish:
                    break
                # Update the step count
                step += 1

                X, y = X.cuda(), y.cuda()
                match_id_batch, O_match_id_batch = None, None
                
                # Use DCE
                if iteration>0 and params.is_DCE:
                    batch_sent_ids = list(range(sentence_id,sentence_id+X.shape[0]))
                    # count the number of entities (not O) in the batch
                    num_samples_batch = np.count_nonzero(np.isin(pos_matrix[:,0],batch_sent_ids))
                    # get the reference feature and the match reference feature
                    match_id_batch = match_id[sample_id*params.top_k:(sample_id+num_samples_batch)*params.top_k]
                    # update count number
                    sample_id += num_samples_batch

                # Use ODCE
                if iteration>0 and params.is_ODCE and len(O_match_id)>0:
                    batch_sent_ids = list(range(sentence_id,sentence_id+X.shape[0]))
                    # count the number of O sampls in the batch
                    num_O_sample_batch = np.count_nonzero(np.isin(O_pos_matrix[:,0],batch_sent_ids))
                    # compute the O_pos_matrix_batch
                    O_pos_matrix_batch = O_pos_matrix[np.isin(O_pos_matrix[:,0],batch_sent_ids)]
                    O_pos_matrix_batch[:,0] = O_pos_matrix_batch[:,0]-sentence_id
                    trainer.O_pos_matrix_batch = O_pos_matrix_batch
                    # get the reference feature and the match reference feature
                    O_match_id_batch = O_match_id[O_sample_id*params.top_k:(O_sample_id+num_O_sample_batch)*params.top_k]
                    # update count number
                    O_sample_id += num_O_sample_batch

                # # For visualization
                # print('------------------------Batch %d------------------------'%step)
                # for i, (sent, sent_y) in enumerate(zip(X,y)):
                #     print('Sent %d-th: %s'%(
                #             i, 
                #             decode_sentence(sent, auto_tokenizer)
                #         ),end='')
                #     for j, word_y in enumerate(sent_y):
                #         if word_y in [-100,0]:
                #             continue
                #         print('')
                #         print(
                #             '%s [%s]'%(
                #                 decode_word_from_sentence(
                #                     X[i],
                #                     j,
                #                     ner_dataloader.auto_tokenizer
                #                 ),
                #                 label_list[word_y]
                #             ),
                #             end='; '
                #         ) 
                #     print('')                     

                # Forward
                trainer.batch_forward(X, 
                                    match_id_batch=match_id_batch,
                                    O_match_id_batch=O_match_id_batch,
                                    max_seq_length=params.max_seq_length)

                # Record training accuracy
                mask_O = torch.not_equal(y, ner_dataloader.O_index)
                mask_pad = torch.not_equal(y, pad_token_label_id)
                eval_mask = torch.logical_and(mask_O, mask_pad)
                predictions = torch.max(trainer.logits,dim=2)[1]
                correct_cnt += int(torch.sum(torch.eq(predictions,y)[eval_mask].float()).item())
                total_cnt += int(torch.sum(eval_mask.float()).item())
                # Compute loss
                if iteration>0:
                    if params.is_distill:
                        ce_loss, distill_loss = trainer.batch_loss_distill(y)
                        ce_list.append(ce_loss)
                        distill_list.append(distill_loss)
                    elif params.is_lucir:
                        ce_loss, distill_loss = trainer.batch_loss_lucir(y)
                        ce_list.append(ce_loss)
                        distill_list.append(distill_loss)
                    elif params.is_podnet:
                        ce_loss, distill_loss = trainer.batch_loss_podnet(y)
                        ce_list.append(ce_loss)
                        distill_list.append(distill_loss)
                    else:
                        ce_loss = trainer.batch_loss(y)
                        ce_list.append(ce_loss)
                else:
                    ce_loss = trainer.batch_loss(y)
                    ce_list.append(ce_loss)
                total_loss = trainer.batch_backward()
                loss_list.append(total_loss)
                mean_loss = np.mean(loss_list)
                mean_distill_loss = np.mean(distill_list) if len(distill_list)>0 else 0
                mean_ce_loss = np.mean(ce_list) if len(ce_list)>0 else 0
                # Update sentence count
                sentence_id += X.shape[0]

                # Print training information
                if params.info_per_steps>0 and step%params.info_per_steps==0:
                    logger.info("Epoch %d, Step %d: Total_loss=%.3f, CE_loss=%.3f, Distill_loss=%.3f, Training_exact_match=%.2f%%"%(
                            e, step, mean_loss, \
                            mean_ce_loss, mean_distill_loss, correct_cnt/total_cnt*100
                    ))
                    # reset the loss lst
                    loss_list = []
                    distill_list = []
                    ce_list = []
                # Update lr + save skpt + do evaluation
                if params.is_train_by_steps:
                    if step>=params.training_steps:
                        is_finish = True
                    # Update learning rate
                    if trainer.scheduler != None:
                        old_lr = trainer.scheduler.get_last_lr()
                        trainer.scheduler.step()
                        new_lr = trainer.scheduler.get_last_lr()
                        if old_lr != new_lr:
                            logger.info("Epoch %d, Step %d: lr is %s"%(
                                e, step, str(new_lr)
                            ))
                    # Save checkpoint 
                    if params.save_per_steps>0 and step%params.save_per_steps==0:
                        trainer.save_model("checkpoint_domain_%s_iteration_%d_steps_%d.pth"%(
                                                domain_name, 
                                                iteration,
                                                step), 
                                            path=params.dump_path)
                    # For evaluation
                    if not params.debug and step%params.evaluate_interval==0:
                        f1_dev, ma_f1_dev, f1_dev_each_class = trainer.evaluate(dataloader_dev, 
                                                                    each_class=True,
                                                                    entity_order=new_entity_list)
                        logger.info("New data: Epoch %d, Step %d: Dev_f1=%.3f, Dev_ma_f1=%.3f, Dev_f1_each_class=%s" % (
                            e, step, f1_dev, ma_f1_dev, str(f1_dev_each_class)
                        ))
                        # f1_dev_cuml, ma_f1_dev_cuml, f1_dev_each_class_cuml = trainer.evaluate(dataloader_dev_cumul, 
                        #                                             each_class=True,
                        #                                             entity_order=all_seen_entity_list)
                        # logger.info("Accumulation: Epoch %d, Step %d: Dev_f1=%.3f, Dev_ma_f1=%.3f, Dev_f1_each_class=%s" % (
                        #     e, step, f1_dev_cuml, ma_f1_dev_cuml, str(f1_dev_each_class_cuml)
                        # ))
                        if f1_dev > best_f1:
                            logger.info("Find better model!!")
                            best_f1 = f1_dev
                            no_improvement_num = 0
                            if iteration==0 and params.is_load_common_first_model:
                                trainer.save_model(common_first_model_ckpt_name, 
                                                    path=os.path.dirname(os.path.dirname(params.dump_path)))
                            else:
                                trainer.save_model(best_model_ckpt_name, path=params.dump_path)
                        else:
                            no_improvement_num += 1
                            logger.info("No better model is found (%d/%d)" % (no_improvement_num, params.early_stop))
                        if no_improvement_num >= params.early_stop:
                            logger.info("Stop training because no better model is found!!!")
                            is_finish = True
            # Check whether mismatching exists
            if iteration>0 and params.is_DCE and not is_finish:
                assert sample_id==num_samples_all, "The sample_id and num_samples_all mismatch!"
                assert sentence_id==num_sentence_all, "The sentence_id and num_sentence_all mismatch!"
                if params.is_ODCE and len(O_match_id)>0:
                    assert O_sample_id==num_O_samples_all, "The O_sample_id and num_O_samples_all mismatch!"

            # Print training information
            if params.info_per_epochs>0 and e%params.info_per_epochs==0:
                logger.info("Epoch %d, Step %d: Total_loss=%.3f, CE_loss=%.3f, Distill_loss=%.3f, Training_exact_match=%.2f%%"%(
                            e, step, mean_loss, \
                            mean_ce_loss, mean_distill_loss, correct_cnt/total_cnt*100
                    ))
            # Update lr + save skpt + do evaluation
            # Update learning rate
            if trainer.scheduler != None:
                old_lr = trainer.scheduler.get_last_lr()
                trainer.scheduler.step()
                new_lr = trainer.scheduler.get_last_lr()
                if old_lr != new_lr:
                    logger.info("Epoch %d, Step %d: lr is %s"%(
                        e, step, str(new_lr)
                    ))
            # Save checkpoint 
            if params.save_per_epochs>0 and e%params.save_per_epochs==0:
                trainer.save_model("checkpoint_domain_%s_iteration_%d_epoch_%d.pth"%(
                                        domain_name, 
                                        iteration,
                                        e), 
                                    path=params.dump_path)
            # For evaluation
            if not params.debug and e%params.evaluate_interval==0:
                f1_dev, ma_f1_dev, f1_dev_each_class = trainer.evaluate(dataloader_dev, 
                                                            each_class=True,
                                                            entity_order=new_entity_list)
                logger.info("New data: Epoch %d, Step %d: Dev_f1=%.3f, Dev_ma_f1=%.3f, Dev_f1_each_class=%s" % (
                    e, step, f1_dev, ma_f1_dev, str(f1_dev_each_class)
                ))
                # f1_dev_cuml, ma_f1_dev_cuml, f1_dev_each_class_cuml = trainer.evaluate(dataloader_dev_cumul, 
                #                                             each_class=True,
                #                                             entity_order=all_seen_entity_list)
                # logger.info("Accumulation: Epoch %d, Step %d: Dev_f1=%.3f, Dev_ma_f1=%.3f, Dev_f1_each_class=%s" % (
                #     e, step, f1_dev_cuml, ma_f1_dev_cuml, str(f1_dev_each_class_cuml)
                # ))
                
                if f1_dev > best_f1:
                    logger.info("Find better model!!")
                    best_f1 = f1_dev
                    no_improvement_num = 0
                    if iteration==0 and params.is_load_common_first_model:
                        trainer.save_model(common_first_model_ckpt_name, 
                                            path=os.path.dirname(os.path.dirname(params.dump_path)))
                    else:
                        trainer.save_model(best_model_ckpt_name, path=params.dump_path)
                else:
                    no_improvement_num += 1
                    logger.info("No better model is found (%d/%d)" % (no_improvement_num, params.early_stop))
                if no_improvement_num >= params.early_stop:
                    logger.info("Stop training because no better model is found!!!")
                    is_finish = True
        # last_global_epoch += 1
        # last_global_step += int(len(dataloader_train.dataset)/params.batch_size)
        logger.info("Finish training ...")

        # ===========================================================================
        # testing
        logger.info("Testing...")
        if params.debug:
            logger.info("Skip testing for debug...")
            continue

        if iteration==0 and params.is_load_common_first_model:
            trainer.load_model(common_first_model_ckpt_name, 
                                path=os.path.dirname(os.path.dirname(params.dump_path)))
        else:
            trainer.load_model(best_model_ckpt_name, path=params.dump_path)
        trainer.model.cuda()

        # Visualization
        # num_seen_labels = 1+class_per_entity*len(all_seen_entity_list)
        # class_center_matrix, features_matrix, targets_list_flatten = \
        #                         compute_class_feature_center(
        #                             dataloader=dataloader_test_cumul,
        #                             feature_model=trainer.model.encoder,
        #                             select_class_indexes=[i for i in range(1,num_seen_labels)],
        #                             is_normalize=True,
        #                             is_return_flatten_feat_and_Y=True
        #                         )
        # plot_distribution(
        #     X=features_matrix, 
        #     Y=targets_list_flatten, 
        #     label_list=label_list[1:num_seen_labels],
        #     sample_ratio=0.9,
        #     select_labels=[11,23,25,27,29,31]
        # )

        # testing
        f1_test_cumul, ma_f1_test_cumul, f1_test_each_class_cumul = trainer.evaluate(dataloader_test_cumul, 
                                                    each_class=True,
                                                    entity_order=all_seen_entity_list,
                                                    is_plot_hist=False)
        logger.info("Accumulation: Test_f1=%.3f, Test_ma_f1=%.3f, Test_f1_each_class=%s"%(
                    f1_test_cumul, ma_f1_test_cumul, str(f1_test_each_class_cumul)))
        logger.info("Finish testing the %d-th iter!"%(iteration+1))

        
def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # random_seed(100)
    params = get_params()
    main_cl(params)
