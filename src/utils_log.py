import os
import pprint
import numpy as np
import argparse
import xlwt
from copy import deepcopy

def get_params():
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="experiments", help="The log path for analyzing")
    parser.add_argument("--output", type=str, default="mean_std", help="Result type(mean_std/mean/max/min/all/std)")
    parser.add_argument("--each_class", action='store_true', help="Get each class result")
    parser.add_argument("--threshold", type=float, default=0, help="The f1 threshold")
    parser.add_argument("--round", type=int, default="2", help="Precision")
    parser.add_argument("--match_cnt", type=int, default=-1, help="Record the i-th matched result (-1 represents the last one)")
    parser.add_argument("--match_str", type=str, default="Test_f1=", help="String before f1")
    parser.add_argument("--match_str_each_class", type=str, default="Test_f1_each_class=", help="String before f1_each_classes")
    parser.add_argument("--is_same_line_each_class", action='store_true', default=True, help="Is the each class result in the same line as the overall result?")
    parser.add_argument("--is_print", action='store_false', default=True, help="If print out the result?")
    params = parser.parse_args()

    ### for debug
    # params.log_path = 'experiments_fg_1_pg_1'
    params.log_path = 'experiments'
    # params.match_str = 'Test_ma_f1='
    # params.match_str_each_class = 'Test_f1_each_class='
    # params.match_cnt = 18
    # params.output = "mean" #"mean_std" #"std" #"mean"
    # params.each_class = True 
    # params.threshold = 0 #
    ###

    return params

domain2entitygroup={
    'ontonotes5': { # #ontonotes5=18 
        'fg_1_pg_1':[['CARDINAL'], ['DATE'], ['EVENT'], ['FAC'], ['GPE'], ['LANGUAGE'], ['LAW'], ['LOC'], ['MONEY'], ['NORP'], ['ORDINAL'], ['ORG'], ['PERCENT'], ['PERSON'], ['PRODUCT'], ['QUANTITY'], ['TIME'], ['WORK_OF_ART']],
        'fg_8_pg_1':[['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC'], ['MONEY'], ['NORP'], ['ORDINAL'], ['ORG'], ['PERCENT'], ['PERSON'], ['PRODUCT'], ['QUANTITY'], ['TIME'], ['WORK_OF_ART']],
        'fg_2_pg_2':[['CARDINAL', 'DATE'], ['EVENT', 'FAC'], ['GPE', 'LANGUAGE'], ['LAW', 'LOC'], ['MONEY', 'NORP'], ['ORDINAL', 'ORG'], ['PERCENT', 'PERSON'], ['PRODUCT', 'QUANTITY'], ['TIME', 'WORK_OF_ART']],
        'fg_8_pg_2':[['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC'], ['MONEY', 'NORP'], ['ORDINAL', 'ORG'], ['PERCENT', 'PERSON'], ['PRODUCT', 'QUANTITY'], ['TIME', 'WORK_OF_ART']],
    },
    'i2b2':{ # #i2b2=16 
        'fg_1_pg_1': [['AGE'], ['CITY'], ['COUNTRY'], ['DATE'], ['DOCTOR'], ['HOSPITAL'], ['IDNUM'], ['MEDICALRECORD'], ['ORGANIZATION'], ['PATIENT'], ['PHONE'], ['PROFESSION'], ['STATE'], ['STREET'], ['USERNAME'], ['ZIP']],
        'fg_8_pg_1': [['AGE', 'CITY', 'COUNTRY', 'DATE', 'DOCTOR', 'HOSPITAL', 'IDNUM', 'MEDICALRECORD'], ['ORGANIZATION'], ['PATIENT'], ['PHONE'], ['PROFESSION'], ['STATE'], ['STREET'], ['USERNAME'], ['ZIP']],
        'fg_2_pg_2': [['AGE', 'CITY'], ['COUNTRY', 'DATE'], ['DOCTOR', 'HOSPITAL'], ['IDNUM', 'MEDICALRECORD'], ['ORGANIZATION', 'PATIENT'], ['PHONE', 'PROFESSION'], ['STATE', 'STREET'], ['USERNAME', 'ZIP']],
        'fg_8_pg_2': [['AGE', 'CITY', 'COUNTRY', 'DATE', 'DOCTOR', 'HOSPITAL', 'IDNUM', 'MEDICALRECORD'], ['ORGANIZATION', 'PATIENT'], ['PHONE', 'PROFESSION'], ['STATE', 'STREET'], ['USERNAME', 'ZIP']],
    },
    'conll2003': { # #conll2003=4
        'fg_1_pg_1': [['location'], ['misc'], ['organisation'], ['person']],
        'fg_2_pg_1': [['location', 'misc'], ['organisation'], ['person']],
    }, 
}

def aggregate_result_dict(result_dict, result_dict_each_class, output, params):
    if output!="all":
        for i in result_dict.keys():
            for j in result_dict[i].keys():
                for k in result_dict[i][j].keys():
                    if output == 'mean':
                        result_dict[i][j][k] = round(np.mean(result_dict[i][j][k]),params.round)
                    elif output == 'max':
                        result_dict[i][j][k] = round(np.max(result_dict[i][j][k]),params.round)
                    elif output == 'min':
                        result_dict[i][j][k] = round(np.min(result_dict[i][j][k]),params.round)
                    elif output == 'std':
                        result_dict[i][j][k] = round(np.std(result_dict[i][j][k]),params.round)
                    else:
                        raise Exception("Invalid output!")

    if params.each_class:
        for i in result_dict_each_class.keys():
            for j in result_dict_each_class[i].keys():
                for k in result_dict_each_class[i][j].keys():
                    f1_dict_all = {}
                    num_f1_dict = len(result_dict_each_class[i][j][k])
                    for f1_dict in result_dict_each_class[i][j][k]:
                        for label, f1_score in f1_dict.items():
                            if output == 'mean':
                                f1_dict_all[label] = round(f1_dict_all.get(label, 0) + f1_score/num_f1_dict,params.round)
                            elif output == 'max':
                                f1_dict_all[label] = round(max(f1_dict_all.get(label, 0),f1_score),params.round)
                            elif output == 'min':
                                f1_dict_all[label] = round(min(f1_dict_all.get(label, 0),f1_score),params.round)
                            elif output == 'std' or output == 'all':
                                tmp_list = f1_dict_all.get(label, [])
                                tmp_list.append(f1_score)
                                f1_dict_all[label] = tmp_list
                            else:
                                raise Exception("Invalid output!")
                    if output == 'std':
                        for label in f1_dict_all.keys():   
                            f1_dict_all[label] = round(np.std(f1_dict_all[label]), params.round)  
                    result_dict_each_class[i][j][k] = f1_dict_all

def main(params):
    result_dict = {}
    result_dict_each_class = {}

    for dir_path_1 in os.listdir(params.log_path):
        exp_name = dir_path_1.split()[-1]
        if exp_name.find('.pth') != -1:
            continue 
        # print(exp_name)
        domain = exp_name.split('_')[0]
        if len(exp_name.split('_'))>1:
            n_sample = exp_name.split('_')[1]
            tmp_idx = len(domain)+len(n_sample)+2
        else:
            n_sample = str(-1)
            tmp_idx = len(domain)+2
        model_name = exp_name[tmp_idx:]
        if os.path.isfile(os.path.join(params.log_path, exp_name)):
            continue
        for dir_path_2 in os.listdir(os.path.join(params.log_path, exp_name)):
            exp_id = dir_path_2.split()[-1]
            # print(exp_id)
            log_file_name = os.path.join(os.path.join(os.path.join(params.log_path, exp_name), exp_id),'train.log')
            with open(log_file_name, encoding='utf-8') as f:
                line = f.readline()
                score = -1
                match_cnt = 0
                match_cnt_each_class = 0
                while line:
                    idx_beg = line.rfind(params.match_str)
                    if idx_beg != -1:  
                        match_cnt += 1
                        idx_end = line.find(',', idx_beg)
                        new_str = line[idx_beg+len(params.match_str):idx_end]
                        if params.match_cnt==-1 or params.match_cnt==match_cnt:
                            score = eval(new_str)
                        if params.is_same_line_each_class and params.each_class:
                            idx_beg = line.rfind(params.match_str_each_class)
                            if idx_beg != -1: 
                                match_cnt_each_class += 1 
                                new_str = line[idx_beg+len(params.match_str_each_class):-1]
                                if params.match_cnt==-1 or params.match_cnt==match_cnt_each_class:
                                    f1_dict = eval(new_str)
                    if (not params.is_same_line_each_class) and params.each_class:
                        idx_beg = line.rfind(params.match_str_each_class)
                        if idx_beg != -1:  
                            match_cnt_each_class += 1
                            new_str = line[idx_beg+len(params.match_str_each_class):-1]
                            if params.match_cnt==-1 or params.match_cnt==match_cnt_each_class:
                                f1_dict = eval(new_str)
                    line = f.readline()
                if score <= params.threshold:
                    continue
                    # raise Exception('No result in file %s'%log_file_name)
                
                if result_dict.get(domain)!=None and \
                        result_dict.get(domain).get(n_sample)!=None and \
                        result_dict.get(domain).get(n_sample).get(model_name)!= None:
                    result_dict[domain][n_sample][model_name].append(score)
                else:
                    if result_dict.get(domain)==None:
                        result_dict[domain] = {}
                    if result_dict.get(domain).get(n_sample)==None:
                        result_dict[domain][n_sample] = {}
                    if result_dict.get(domain).get(n_sample).get(model_name)==None:
                        result_dict[domain][n_sample][model_name] = [score]

                if params.each_class:
                    if result_dict_each_class.get(domain)!=None and \
                            result_dict_each_class.get(domain).get(n_sample)!=None and \
                            result_dict_each_class.get(domain).get(n_sample).get(model_name)!= None:
                        result_dict_each_class[domain][n_sample][model_name].append(f1_dict)
                    else:
                        if result_dict_each_class.get(domain)==None:
                            result_dict_each_class[domain] = {}
                        if result_dict_each_class.get(domain).get(n_sample)==None:
                            result_dict_each_class[domain][n_sample] = {}
                        if result_dict_each_class.get(domain).get(n_sample).get(model_name)==None:
                            result_dict_each_class[domain][n_sample][model_name] = [f1_dict]

    if params.output=='mean_std':  
        
        # mean
        mean_result_dict = deepcopy(result_dict)
        mean_result_dict_each_class = deepcopy(result_dict_each_class)
        aggregate_result_dict(mean_result_dict, mean_result_dict_each_class, 'mean', params)
        # std
        std_result_dict = deepcopy(result_dict)
        std_result_dict_each_class = deepcopy(result_dict_each_class)
        aggregate_result_dict(std_result_dict, std_result_dict_each_class, 'std', params)
        # concat the result with '±'
        for i in result_dict.keys():
            for j in result_dict[i].keys():
                for k in result_dict[i][j].keys():
                    result_dict[i][j][k] = str(mean_result_dict[i][j][k])+'±'+ \
                                            str(std_result_dict[i][j][k])
        for i in result_dict_each_class.keys():
            for j in result_dict_each_class[i].keys():
                for k in result_dict_each_class[i][j].keys():
                    if len(result_dict_each_class[i][j][k])==0:
                        continue
                    label_list = result_dict_each_class[i][j][k][0].keys()
                    result_dict_each_class[i][j][k] = {}
                    for label in label_list:
                        result_dict_each_class[i][j][k][label] = str(mean_result_dict_each_class[i][j][k][label])+'±'+ \
                                            str(std_result_dict_each_class[i][j][k][label])
    else:
        aggregate_result_dict(result_dict, result_dict_each_class, params.output, params)
  
    if params.each_class:
        if params.is_print:
            pprint.pprint(result_dict)
            pprint.pprint(result_dict_each_class)
        return result_dict, result_dict_each_class
    else:
        if params.is_print:
            pprint.pprint(result_dict)
        return result_dict

def flatten_nested_dict(nested_dict):
    key_list = []
    flatten_dict = {}
    for k1,v1 in nested_dict.items():
        for k2,v2 in nested_dict[k1].items():
            key_list.append(k1+'_'+k2)
            flatten_dict[k1+'_'+k2] = nested_dict[k1][k2]
    return flatten_dict

def get_mean_std(f1_result):
    if f1_result.find('±')==-1:
        f1_mean, f1_std = eval(f1_result), 0.0
    else:
        f1_mean, f1_std = eval(f1_result[:f1_result.find('±')]), \
                            eval(f1_result[f1_result.find('±')+1:])
    return f1_mean, f1_std

def get_entity_mean_f1(result_dict_each_class, entity_list):
    f1_list = []
    for e in entity_list:
        f1_result = result_dict_each_class[e]        
        f1_mean, f1_std = get_mean_std(f1_result)
        f1_list.append(f1_mean)
    return np.mean(f1_list)

def convert_2_excel():

    params = get_params()
    setting_name = params.log_path.split('_',1)[1]

    match_str_list = ['Test_f1=','Test_ma_f1=','Forget']
    # get domain list
    params.each_class = True
    params.is_print = False
    params.match_cnt = 1
    result_dict, _ = main(params)
    domain_list = list(result_dict.keys())

    # each domain corresponds to a table
    workbook=xlwt.Workbook(encoding='utf-8')
    for domain in domain_list:
        booksheet=workbook.add_sheet(domain, cell_overwrite_ok=True)
        style = xlwt.XFStyle()
        alignment = xlwt.Alignment()
        alignment.horz = xlwt.Alignment.HORZ_CENTER
        alignment.vert = xlwt.Alignment.VERT_CENTER
        style.alignment = alignment
        # initialize the aggregate result
        method_aggre_result = {}
        for match_str in match_str_list:
            method_aggre_result[match_str] = {}
        # initialize the macro_f1_matrix to compute Forget
        num_step = len(domain2entitygroup[domain][setting_name])
        # for each method corresponds to a num_step*num_step matrix
        # ================================
        #         | task1 | task2 | task3
        # ================================
        # step 1  |       |   /   |   /
        # ================================
        # step 2  |       |       |   /
        # ================================
        # step 3  |       |       |
        # ================================
        method_step_task_f1 = {}
        # each step
        match_cnt = 1
        is_finish = False
        while True:
            # micro/macro f1
            for match_str_idx, match_str in enumerate(match_str_list):
                col_idx = len(match_str_list)*(match_cnt-1)+match_str_idx+1
                # get result_dict and method_list
                if match_cnt==1 and match_str=='Test_f1=':
                    params.match_cnt = match_cnt
                    params.match_str = match_str
                    result_dict, result_dict_each_class = main(params)
                    flatten_result_dict = flatten_nested_dict(result_dict[domain])
                    method_list=sorted(list(flatten_result_dict.keys()))
                    for idx, method in enumerate(method_list):
                        booksheet.write(1+idx,0,method,style)
                # get result_dict
                elif match_str!= 'Forget':
                    params.match_cnt = match_cnt
                    params.match_str = match_str
                    result_dict, result_dict_each_class = main(params)
                    flatten_result_dict = {}
                    if domain in result_dict.keys():
                        flatten_result_dict = flatten_nested_dict(result_dict[domain])
                    if len(flatten_result_dict.keys())==0:
                        is_finish = True
                        break
                # compute Forget
                else:
                    assert match_cnt<=num_step, "match_cnt > num_step !!!"
                    # compute forget for each method
                    flatten_result_dict_each_class = flatten_nested_dict(result_dict_each_class[domain])
                    for method in method_list:
                        if method in flatten_result_dict_each_class.keys():
                            if match_cnt==1:
                                method_step_task_f1[method] = np.zeros((num_step,num_step))
                            for step_idx in range(match_cnt):
                                method_step_task_f1[method][match_cnt-1,step_idx] = \
                                    get_entity_mean_f1(flatten_result_dict_each_class[method], domain2entitygroup[domain][setting_name][step_idx])         
                    
                # write into the excel
                if match_str_idx==0:
                    booksheet.write_merge(0,0,col_idx,col_idx+len(match_str_list)-1,'Step %d'%match_cnt,style)
                for idx, method in enumerate(method_list):
                    row_idx = 1+idx
                    # write the mean f1 into excel
                    if match_str!='Forget':
                        if method in flatten_result_dict.keys():
                            f1_result = flatten_result_dict[method]        
                            f1_mean, f1_std = get_mean_std(f1_result)
                            booksheet.write(row_idx,col_idx,f1_result,style)
                            # aggregate the result
                            if method not in method_aggre_result[match_str].keys():
                                method_aggre_result[match_str][method] = []
                            method_aggre_result[match_str][method].append(f1_mean)
                    # write the Forget into excel
                    else:
                        if method in flatten_result_dict_each_class.keys():
                            forgetting = 0.0
                            if match_cnt==1:
                                booksheet.write(row_idx,col_idx,forgetting,style)
                                continue
                            for idx_1 in range(match_cnt-1):
                                task_forgetting = -1000
                                for idx_2 in range(match_cnt-1):
                                    if idx_2<idx_1:
                                        continue
                                    tmp = method_step_task_f1[method][idx_2,idx_1] - \
                                        method_step_task_f1[method][match_cnt-1,idx_1]
                                    if tmp>task_forgetting:
                                        task_forgetting=tmp
                                forgetting += task_forgetting
                            forgetting = forgetting/(match_cnt-1)
                            booksheet.write(row_idx,col_idx,'%.2f'%forgetting,style)
                            # aggregate the result
                            if method not in method_aggre_result[match_str].keys():
                                method_aggre_result[match_str][method] = []
                            method_aggre_result[match_str][method].append(forgetting)

            if is_finish:
                break
            match_cnt += 1

        for i, match_str in enumerate(match_str_list):
            col_idx = len(match_str_list)*(match_cnt-1)+i+1
            if i==0:
                booksheet.write_merge(0,0,col_idx,col_idx+len(match_str_list)-1,'average_result',style)
            for j, method in enumerate(method_list):
                if method in method_aggre_result[match_str].keys() and len(method_aggre_result[match_str][method])>0:
                    average_res = np.mean(method_aggre_result[match_str][method])
                    booksheet.write(j+1,col_idx,'%.2f'%average_res,style)

    # save as excel format
    workbook.save('%s\\results_%s.xlsx'%(params.log_path,params.log_path))

if __name__ == "__main__":
    params = get_params()
    main(params)
    # convert_2_excel()