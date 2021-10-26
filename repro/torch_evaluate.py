
import os, sys
from operator import itemgetter
from collections import Counter
from tqdm import tqdm
from pprint import pprint

from sklearn.metrics import classification_report
from scipy.special import softmax

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification

from torch_dataloader import load_data
from utils import d_score, average, var, avg2dict

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = 4,
)
model.cuda()

d_list=[]

def torch_eval(test_dataloader, o):
    model.eval()
    predictions, true_labels = [], []
    sconj_type_list = []
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels, b_sconj = batch
        with torch.no_grad():
        # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        sconj_ids = b_sconj.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)
        sconj_type_list.append(sconj_ids)
    #print('Prediction DONE.')

    y_true_multi = []
    y_pred_multi = []
    for i in range(len(true_labels)):
        true_labels_i = true_labels[i]
        y_true_multi.append(true_labels_i[0])
        #print(predictions[i])
        s_max = softmax(predictions[i])
        #print(s_max)
        if o == True:
            s_max = restrict(s_max, int(sconj_type_list[i][0]))
        #print(s_max)
        pred_labels_i = np.argmax(s_max, axis=1).flatten()#predictions[i]
        y_pred_multi.append(pred_labels_i[0])

    d = classification_report(y_true_multi, y_pred_multi, output_dict=True)
    return d

s_type=[]
def restrict(s_max, sconj_type):
    new_s_max= s_max
    s_type.append(sconj_type)
    if int(sconj_type) == 1: ##since,as
        new_s_max[0][1] = 0 #cond=0
        new_s_max[0][2] = 0 #conc=0
    elif  int(sconj_type) == 2: ##while
        new_s_max[0][0] = 0 #cause=0
        new_s_max[0][1] = 0 #cond=0
    elif int(sconj_type) == 3: ##when
        new_s_max[0][0] = 0 #cause=0
        new_s_max[0][2] = 0 #conc=0
    else: return new_s_max
    #print(new_s_max)
    return new_s_max

def softmax(x):
    u = np.sum(np.exp(x))
    return np.exp(x)/u

def torch_predict(output_name, mode, avg, method, o, r):

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
            print('Test Split: '+test_path)
            model_path = os.path.join(output_name,str(cv_count)+'.pth')
            print('Model: '+model_path)

            test_dataloader = load_data(test_path, s=False, batch_size=1)
            model.load_state_dict(torch.load(model_path))
            d = torch_eval(test_dataloader, o)
            d_list.append(d)
    
    elif method == 'MIX_CV':
        eval_path = '../rsc/amr/TEST'

        for id_count in range(1,6):
            for cv_count in range(1,6):
                test_path = eval_path+str(cv_count)+'.csv'
                print('Test Split: '+test_path)
                model_path = os.path.join(output_name,str(r)+'_'+str(id_count)+'_10_5e-05_16',str(cv_count)+'.pth')
                print('Model: '+model_path)

                test_dataloader = load_data(test_path, s=False, batch_size=1)
                model.load_state_dict(torch.load(model_path))
                d = torch_eval(test_dataloader, o)
                d_list.append(d)

    elif method == 'SEED':

        if mode == 'dev':
            eval_path = '../rsc/wiki/merged/eval_dummy.csv'
        elif mode == 'test':
            eval_path = '../rsc/amr/TEST'
        else:
            print('---Mode NOT Specified---')
            return

        if mode == 'dev':
            for seed_num in range(5):
                test_path = eval_path
                print(test_path)
                model_path = os.path.join(output_name,str(seed_num)+'.pth')
                print(model_path)

                test_dataloader = load_data(test_path, batch_size=64)
                model.load_state_dict(torch.load(model_path))
                d = torch_eval(test_dataloader, o)
                d_list.append(d)

        elif mode == 'test':
            eval_path = '../rsc/amr/TEST'

            if avg == 'simple':
                for seed_num in range(5):
                    for cv_count in range(1,6):
                        test_path = eval_path+str(cv_count)+'.csv'
                        print('Test Split: '+test_path)
                        model_path = os.path.join(output_name,str(seed_num)+'.pth')
                        print('Model: '+model_path)

                        test_dataloader = load_data(test_path, s=False, batch_size=1)
                        model.load_state_dict(torch.load(model_path))
                        d = torch_eval(test_dataloader, o)
                        d_list.append(d)
            
            else:
                print('---Mode NOT Specified---')
                sys.exit()

    else:
        if mode == 'pdtb':
            cv_count=0
            test_path = '../rsc/pdtb/'+method+'.csv'
            model_path = os.path.join(output_name,str(cv_count)+'.pth')
            test_dataloader = load_data(test_path, s=False, batch_size=1)
            model.load_state_dict(torch.load(model_path))
            d = torch_eval(test_dataloader, o)
            d_list.append(d)

def torch_find_scores(rd, output_name, mode, avg, method, o=False, r=0):
    torch_predict(output_name, mode, avg, method, o, r)
    scores = []
    for d in d_list:
        pprint(d)
        s = d_score(d)
        scores.append(s)
    if mode == 'pdtb':
        pprint(scores)
        return

    avg = average(scores, d_list)
    final_d = avg2dict(rd, avg)

    if mode == 'dev':
        micro = final_d['micro']
        result = micro
    elif mode == 'test':
        print('---Averaged Scores---')
        result = final_d
        pprint(final_d)
        print('---Variances---')
        var_dict = var(rd, scores, d_list)
        pprint(var_dict)
    else:
        print('---Mode NOT Specified---')
        return
    return result

def torch_find_best(rd, dir_name, avg, method):
    dir_name = os.path.join('torch_models',dir_name)
    results = []
    models = os.listdir(dir_name)
    for model in tqdm(models):
        output_name = os.path.join(dir_name,model)
        print('---Evaluating: '+str(output_name)+'---')
        micro = torch_find_scores(rd, output_name, 'dev', avg, method)
        result = [model, micro]
        print('---micro: '+str(result[1])+'---')
        results.append(result)
    results.sort(key=itemgetter(1), reverse=True)
    return results

def check_predictions(rd, output_name, mode, avg, method, o, r=0):
    if mode == 'check':

        eval_path = '../rsc/amr/TEST'
        cv_count=0
        test_path = eval_path+str(cv_count)+'.csv'
        print(test_path)
        model_path = os.path.join(output_name,str(cv_count)+'.pth')
        print(model_path)

        test_dataloader = load_data(test_path, batch_size=1)
        model.load_state_dict(torch.load(model_path))
        t, p_f, d_f = torch_check(test_dataloader, o=False)
        print('TRUE')
        print(t)
        print('---')
        print('PRED_original')
        print(p_f)
        t, p_t, d_t = torch_check(test_dataloader, o=True)
        print('PRED_restriction')
        print(p_t)
        print('---')
        print('REPORT_original')
        pprint(d_f)
        print('REPORT_restriction')
        pprint(d_t)
        wrong_f = []
        f_i = []
        wrong_t = []
        t_i = []
        for i in range(len(t)):
            if p_f[i] != t[i]:
                result_f = str(i)+': '+str(t[i])+'->'+str(p_f[i])
                f_i.append(i)
                wrong_f.append(result_f)
            if p_t[i] != t[i]:
                result_t = str(i)+': '+str(t[i])+'->'+str(p_t[i])
                t_i.append(i)
                wrong_t.append(result_t)
        diff = []
        for i in f_i:
            if i not in t_i:
                diff.append(i)
        print(wrong_f)
        print(wrong_t)
        print(diff)

        s_dict={1:[],2:[],3:[]}
        print(s_type)
        for i in range(len(s_type)):
            if s_type[i] != 0:
                s_dict[s_type[i]].append(i)

        print(s_dict)

        pred_dict={0:[],1:[],2:[],3:[]}
        for i in range(len(p_f)):
            pred_dict[t[i]].append(p_f[i])

        c0,c1,c2,c3 = Counter(pred_dict[0]),Counter(pred_dict[1]),Counter(pred_dict[2]),Counter(pred_dict[3])
        print(c0)
        print(c1)
        print(c2)
        print(c3)
        #torch_predict(output_name, 'check', avg, method, o, r)
    else:
        print('---Mode NOT Specified---')

def torch_check(test_dataloader, o):
    model.eval()
    predictions, true_labels = [], []
    sconj_type_list = []
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels, b_sconj = batch
        with torch.no_grad():
        # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        sconj_ids = b_sconj.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)
        sconj_type_list.append(sconj_ids)
    print('Prediction DONE.')

    y_true_multi = []
    y_pred_multi = []
    for i in range(len(true_labels)):
        true_labels_i = true_labels[i]
        y_true_multi.append(true_labels_i[0])
        s_max = softmax(predictions[i])
        if o == True:
            s_max = restrict(s_max, int(sconj_type_list[i][0]))
        pred_labels_i = np.argmax(s_max, axis=1).flatten()
        y_pred_multi.append(pred_labels_i[0])
    
    d = classification_report(y_true_multi, y_pred_multi, output_dict=True)

    return y_true_multi, y_pred_multi, d

