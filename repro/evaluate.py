from simpletransformers.classification import ClassificationModel
from pprint import pprint
import numpy as np
import csv
import os
from operator import itemgetter
from sklearn.metrics import classification_report
from scipy.special import softmax

SAVE_DIR = 'simple_models'

d_list = []

def predict(output_name, mode, avg, method):

    if method == 'CV':

        if mode == 'dev':
            eval_path = '../rsc/amr/DEV'
        elif mode == 'test':
            eval_path = '../rsc/amr/TEST'
        else:
            print('---Mode NOT Specified---')
            return

        for cv_count in range(1,6):
            test_path = eval_path+str(cv_count)+'.csv'
            print(test_path)
            output = os.path.join(SAVE_DIR, output_name,str(cv_count))
            print(output)
            test_data = []
            y_true_multi = []

            with open(test_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    test_data.append(row[:2])
                    y_true_multi.append(int(row[2]))

            model = ClassificationModel('bert', output, args={})
            y_pred_multi, _ = model.predict(test_data)
            d = classification_report(y_true_multi, y_pred_multi,output_dict=True)
            d_list.append(d)
    
    elif method == 'SEED':

        if mode == 'dev':
            eval_path = '../../rsc/wiki/merged/eval.csv'

            for seed_num in range(5):
                test_path = eval_path
                output = os.path.join(SAVE_DIR, output_name,str(seed_num))
                test_data = []
                y_true_multi = []

                with open(test_path, mode='r', encoding='utf-8') as f:
                    reader = csv.reader(f, delimiter='\t')
                    for row in reader:
                        test_data.append(row[:2])
                        y_true_multi.append(int(row[2]))

                model = ClassificationModel('bert', output, args={})
                y_pred_multi, _ = model.predict(test_data)
                d = classification_report(y_true_multi, y_pred_multi,output_dict=True)
                d_list.append(d)

        elif mode == 'test':
            eval_path = '/home/cl/yuki-yama/phd/IWCS/rsc/amr/TEST'

            if avg == 'simple':
                for seed_num in range(5):
                    for cv_count in range(1,6):
                        test_path = eval_path+str(cv_count)+'.csv'
                        print(test_path)
                        output = os.path.join(SAVE_DIR,output_name,str(seed_num))
                        test_data = []
                        y_true_multi = []

                        with open(test_path, mode='r', encoding='utf-8') as f:
                            reader = csv.reader(f, delimiter='\t')
                            for row in reader:
                                test_data.append(row[:2])
                                y_true_multi.append(int(row[2]))

                        model = ClassificationModel('bert', output, args={})
                        y_pred_multi, _ = model.predict(test_data)
                        d = classification_report(y_true_multi, y_pred_multi,output_dict=True)
                        d_list.append(d)
            
            elif avg == 'soft-vote':
                for cv_count in range(1,6):
                    test_path = eval_path+str(cv_count)+'.csv'
                    print(test_path)
                    test_data = []
                    y_true_multi = []

                    with open(test_path, mode='r', encoding='utf-8') as f:
                        reader = csv.reader(f, delimiter='\t')
                        for row in reader:
                            test_data.append(row[:2])
                            y_true_multi.append(int(row[2]))

                    model = ClassificationModel('bert', os.path.join(SAVE_DIR,output_name,str(0)), args={})
                    _, raw_outputs_0 = model.predict(test_data)
                    model = ClassificationModel('bert', os.path.join(SAVE_DIR,output_name,str(1)), args={})
                    _, raw_outputs_1 = model.predict(test_data)
                    model = ClassificationModel('bert', os.path.join(SAVE_DIR,output_name,str(2)), args={})
                    _, raw_outputs_2 = model.predict(test_data)
                    model = ClassificationModel('bert', os.path.join(SAVE_DIR,output_name,str(3)), args={})
                    _, raw_outputs_3 = model.predict(test_data)
                    model = ClassificationModel('bert', os.path.join(SAVE_DIR,output_name,str(4)), args={})
                    _, raw_outputs_4 = model.predict(test_data)

                    y_pred_ave = (softmax(raw_outputs_0, axis=1)+softmax(raw_outputs_1, axis=1)+softmax(raw_outputs_2, axis=1)+softmax(raw_outputs_3, axis=1)+softmax(raw_outputs_4, axis=1))/5
                    soft_vote = [np.argmax(i) for i in y_pred_ave]
                    y_pred_multi = soft_vote

                    y_pred_multi, _ = model.predict(test_data)
                    d = classification_report(y_true_multi, y_pred_multi, output_dict=True)
                    d_list.append(d)
            
            else: print('---Mode NOT Specified---')
    
    else: print('NO Method Indicated')

    return

def d_score(d):
    s = []  
    for i in range(4):
        label = str(i)
        s.append(d[label]['precision'])
        s.append(d[label]['recall'])
        s.append(d[label]['f1-score'])
    s.append(d['macro avg']['precision'])
    s.append(d['macro avg']['recall'])
    s.append(d['macro avg']['f1-score'])
    s.append(d['accuracy'])
    return s

def average(scores, d_list):
    total = np.zeros(16)
    for s in scores:
        s = np.array(s)
        total += s
    avg_score = total/int(len(d_list))
    return avg_score

'''
Variance
'''
def var(rd, scores, d_list):
    p0,r0,f0 = [],[],[]
    p1,r1,f1 = [],[],[]
    p2,r2,f2 = [],[],[]
    p3,r3,f3 = [],[],[]
    mp,mr,mf,a = [],[],[],[]
    var_list = []
    var_label_list = []
    for s in scores:
        p0.append(s[0])
        r0.append(s[1])
        f0.append(s[2])
        p1.append(s[3])
        r1.append(s[4])
        f1.append(s[5])
        p2.append(s[6])
        r2.append(s[7])
        f2.append(s[8])
        p3.append(s[9])
        r3.append(s[10])
        f3.append(s[11])          
        mp.append(s[12])
        mr.append(s[13])
        mf.append(s[14])
        a.append(s[15])
    var_label_list.append(np.var(p0))
    var_label_list.append(np.var(r0))
    var_label_list.append(np.var(f0))
    var_label_list.append(np.var(p1))
    var_label_list.append(np.var(r1))
    var_label_list.append(np.var(f1))
    var_label_list.append(np.var(p2))
    var_label_list.append(np.var(r2))
    var_label_list.append(np.var(f2))
    var_label_list.append(np.var(p3))
    var_label_list.append(np.var(r3))
    var_label_list.append(np.var(f3))
    var_mp = np.var(mp)
    var_list.append(var_mp)
    var_mr = np.var(mr)
    var_list.append(var_mr)
    var_mf = np.var(mf)
    var_list.append(var_mf)
    var_a = np.var(a)
    var_list.append(var_a)

    var_dict = {'macro': {'precision': 0,'recall': 0, 'f1-score': 0}, 'micro': 0, '0': {'precision': 0,'recall': 0, 'f1-score': 0}, '1': {'precision': 0,'recall': 0, 'f1-score': 0}, '2': {'precision': 0,'recall': 0, 'f1-score': 0}, '3': {'precision': 0,'recall': 0, 'f1-score': 0}}
    
    if rd == True:
        var_list = [round(f, 2) for f in var_list*np.array([100])/int(len(d_list))]
    else: var_list = [f for f in var_list*np.array([100])/int(len(d_list))]
    var_dict['macro']['precision'] = var_list[0]
    var_dict['macro']['recall'] = var_list[1]
    var_dict['macro']['f1-score'] = var_list[2]
    var_dict['micro'] = var_list[3]

    if rd == True:
        var_label_list = [round(f, 2) for f in var_label_list*np.array([100])/int(len(d_list))]
    else: var_label_list = [f for f in var_label_list*np.array([100])/int(len(d_list))]
    var_dict['0']['precision'] = var_label_list[0]
    var_dict['0']['recall'] = var_label_list[1]
    var_dict['0']['f1-score'] = var_label_list[2]
    var_dict['1']['precision'] = var_label_list[3]
    var_dict['1']['recall'] = var_label_list[4]
    var_dict['1']['f1-score'] = var_label_list[5]
    var_dict['2']['precision'] = var_label_list[6]
    var_dict['2']['recall'] = var_label_list[7]
    var_dict['2']['f1-score'] = var_label_list[8]
    var_dict['3']['precision'] = var_label_list[9]
    var_dict['3']['recall'] = var_label_list[10]
    var_dict['3']['f1-score'] = var_label_list[11]

    return var_dict

'''
Simple Average
'''
def avg2dict(rd, avg):
    if rd == True:
        avg = [round(f, 2) for f in avg*100]
    final_d = {'0': {'precision': 0,'recall': 0,'f1-score': 0},\
        '1': {'precision': 0,'recall': 0,'f1-score': 0},\
            '2': {'precision': 0,'recall': 0,'f1-score': 0},\
                '3': {'precision': 0,'recall': 0,'f1-score': 0},\
                    'macro': {'precision': 0,'recall': 0,'f1-score': 0},'micro': 0}
    final_d['0']['precision'] = avg[0]
    final_d['0']['recall'] = avg[1]
    final_d['0']['f1-score'] = avg[2]
    final_d['1']['precision'] = avg[3]
    final_d['1']['recall'] = avg[4]
    final_d['1']['f1-score'] = avg[5]
    final_d['2']['precision'] = avg[6]
    final_d['2']['recall'] = avg[7]
    final_d['2']['f1-score'] = avg[8]
    final_d['3']['precision'] = avg[9]
    final_d['3']['recall'] = avg[10]
    final_d['3']['f1-score'] = avg[11]
    final_d['macro']['precision'] = avg[12] 
    final_d['macro']['recall'] = avg[13]
    final_d['macro']['f1-score'] = avg[14]
    final_d['micro'] = avg[15]
    return final_d

def find_scores(rd, output_name, mode, avg, method):
    predict(output_name, mode, avg, method)
    scores = []
    for d in d_list:
        s = d_score(d)
        scores.append(s)
    avg = average(scores, d_list)
    final_d = avg2dict(rd, avg)
    #pprint(final_d)

    if mode == 'dev':
        micro = final_d['micro']
        result = micro
    elif mode == 'test':
        result = final_d
        pprint(final_d)
        var_dict = var(rd, scores, d_list)
        pprint(var_dict)

    else: print('---Mode NOT Specified---')
    return result

def find_best(rd, dir_name, avg, method):
    dir_name = os.path.join(SAVE_DIR,dir_name)
    results = []
    models = os.listdir(dir_name)
    for model in models:
        output_name = os.path.join(dir_name,model)
        print('---Evaluating: '+str(output_name)+'---')
        micro = find_scores(rd, output_name, 'dev', avg, method)
        result = [model, micro]
        print('---micro: '+str(result[1])+'---')
        results.append(result)
    results.sort(key=itemgetter(1), reverse=True)
    return results
