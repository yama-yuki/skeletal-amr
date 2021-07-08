'''
## TRAIN
## CV BERT-AMR
CUDA_VISIBLE_DEVICES=0 nohup python main.py -m train --data amr -e 10 -l 5e-05 -b 16 > cv.out &
## CV WIKI-AMR
CUDA_VISIBLE_DEVICES=0 nohup python main.py -m train --data amr --target BERT-WIKI/3_2e-05_64/0 -e 10 -l 3e-05 -b 32 > cv.out &
## FULL BERT-AMR
CUDA_VISIBLE_DEVICES=0 nohup python main.py -m train --data amr -e 10 -l 5e-05 -b 16 -f > full.out &
## FULL WIKI-AMR
CUDA_VISIBLE_DEVICES=0 nohup python main.py -m train --data amr --target BERT-WIKI/3_2e-05_64/0 -e 10 -l 3e-05 -b 32 -f > full.out &

## TEST
CUDA_VISIBLE_DEVICES=0 python main.py -m test -n BERT-AMR/10_5e-05_16 -r -rs
CUDA_VISIBLE_DEVICES=0 python main.py -m test -n BERT-WIKI/3_2e-05_64 -r -rs

## PDTB
CUDA_VISIBLE_DEVICES=0 python main.py -m pdtb -n BERT-AMR/10_5e-05_16 -r
CUDA_VISIBLE_DEVICES=0 python main.py -m pdtb -n WIKI-AMR/WIKI_3_2e-05_64_AMR_10_3e-05_32 -r

args.modelname
results = torch_find_scores(args.rounding, output_name, args.mode, avg='simple', method='CV', args.restrict_softmax)


'''

import os
from tqdm import tqdm

from torch_train import torch_train
from torch_evaluate import torch_find_scores, torch_find_best, check_predictions

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-m','--mode',choices=['train','dev','test','check','pdtb'],help="select a mode")
parser.add_argument('--data',choices=['amr','wiki','mix'],help="data for finetuning",type=str)
parser.add_argument("-s","--sconj",help="True: train with sconj, False: train without sconj",action='store_true',default=False)
parser.add_argument("-f","--full",help="True: train a full model, False: train 5 splits for cv",action='store_true',default=False)

## Model Names
parser.add_argument("-d","--dirname",help="model dir name: BERT-AMR",type=str,default=None)
parser.add_argument("-n","--modelname",help="model name: BERT-AMR/3_2e-05_16",type=str)
parser.add_argument("-t","--target",help="specific model name: BERT-WIKI/3_2e-05_64/0",type=str)

## For MIX models
parser.add_argument("--mix",help="For MIX models: 1-5",type=int,default=1)
parser.add_argument("--mixid",help="For MIX models: 1-25",type=int,default=1)

## Training Hyperparameter
parser.add_argument("-e","--epoch",help="(TRAIN) epochs:3,5,10",type=int)
parser.add_argument("-l","--lr",help="(TRAIN) learning_rate:2e-5,3e-5,5e-5",type=float)
parser.add_argument("-b","--batch",help="(TRAIN) batch_size:16,32,64",type=int)

## Evaluation
parser.add_argument("-r","--rounding",help="(TEST) round-off: True",action='store_true')
parser.add_argument("-a","--averaging",help="(TEST) simple,soft-vote",default='simple',type=str)
parser.add_argument("-rs","--restrict_softmax",help="(TEST) Restriction: True, Vanilla: False",action='store_true',default=False)

args = parser.parse_args()

model_A = ['BERT-AMR', 'BERT-AMR_tag', 'BERT-AMR_notag', 'BERT-AMRs', 'BERT-MIX', 'BERT-MIXs', 'WIKI-AMR', 'WIKI-AMRs', 'WIKI-MIX']
model_W = ['BERT-WIKI', 'BERT-WIKIs']

def main():

    ## TRAIN
    if args.mode == 'train':
        if args.data == 'amr':
            if args.target:
                print('WIKI-AMR') ## BERT-WIKI-AMR
                model_name = str(args.epoch)+'_'+str(args.lr)+'_'+str(args.batch)
                print(model_name)
                if args.full == True:
                    cv_num = 0
                    torch_train(model_name, args.data, args.sconj, args.epoch, args.lr, args.batch, cv_num, model_to_tune=args.target)
                else:
                    for cv_num in tqdm(range(1,6)):
                        torch_train(model_name, args.data, args.sconj, args.epoch, args.lr, args.batch, cv_num, model_to_tune=args.target)

            else:
                print('BERT-AMR')
                model_name = str(args.epoch)+'_'+str(args.lr)+'_'+str(args.batch)
                print(model_name)
                if args.full == True:
                    cv_num = 0
                    torch_train(model_name, args.data, args.sconj, args.epoch, args.lr, args.batch, cv_num)
                else:
                    for cv_num in tqdm(range(1,6)):
                        torch_train(model_name, args.data, args.sconj, args.epoch, args.lr, args.batch, cv_num)

        elif args.data == 'mix':
            if args.target:
                print('WIKI-MIX') ## BERT-WIKI-MIX
                model_name = str(args.epoch)+'_'+str(args.lr)+'_'+str(args.batch)
                print(model_name)
                if args.full == True:
                    pass
                else:
                    for cv_num in tqdm(range(1,6)):
                        torch_train(model_name, args.data, args.sconj, args.epoch, args.lr, args.batch, cv_num, model_to_tune=args.target)

            else:
                print('BERT-MIX')
                model_name = str(args.epoch)+'_'+str(args.lr)+'_'+str(args.batch)
                print(model_name)
                if args.full == True:
                    pass
                else:
                    for cv_num in tqdm(range(1,6)):
                        torch_train(model_name, args.data, args.sconj, args.epoch, args.lr, args.batch, cv_num, args.mix, args.mixid)

        elif args.data == 'wiki':
            print('---Train on WIKI---')
            print('BERT-WIKI')
            model_name = str(args.epoch)+'_'+str(args.lr)+'_'+str(args.batch)
            print(model_name)
            for seed_num in tqdm(range(5)):
                torch_train(model_name, args.data, args.sconj, args.epoch, args.lr, args.batch, seed_num)
        print('---Done Training---')

    ## DEV for Grid Search
    elif args.mode == 'dev':
        #args.modelname: BERT-AMR/3_3e-05_16
        if args.modelname:
            print('---Evaluate on DEV---')
            if args.modelname.split('/')[0] in model_A:
                output_name = os.path.join('torch_models',args.modelname)
                method = 'CV'
                avg = 'simple'
                micro = torch_find_scores(args.rounding, output_name, args.mode, avg, method)
                print('micro-f1: '+str(micro))
            elif args.modelname.split('/')[0] in model_W:
                output_name = os.path.join('torch_models',args.modelname)
                method = 'SEED'
                micro = torch_find_scores(args.rounding, output_name, args.mode, args.averaging, method)
                print('micro-f1: '+str(micro))
        
        elif args.dirname:
            #args.dirname: BERT-AMR
            print('---Find Best Model on DEV---')
            if args.dirname in model_A:
                results = torch_find_best(args.rounding, args.dirname, avg='simple', method='CV')
                print('---Score Ranking: '+str(args.dirname)+'---')
                for model, score in results:
                    print(model+': '+str(score))
            elif args.dirname in model_W:
                results = torch_find_best(args.rounding, args.dirname, args.averaging, method='SEED')
                print('---Score Ranking: '+str(args.dirname)+'---')
                for model, score in results:
                    print(model+': '+str(score))

    ## TEST
    elif args.mode == 'test':
        print('---Evaluate on TEST---')
        if args.dirname == 'BERT-MIX':
            ## SEED & CV
            output_name = os.path.join('../torch_models', args.dirname)
            avg = 'simple'
            method = 'MIX_CV'
            results = torch_find_scores(args.rounding, output_name, args.mode, avg, method, args.restrict_softmax, args.mix)

        else:
            output_name = os.path.join('../torch_models', args.modelname)
            print('Model Dir: '+output_name+'\n')
            if args.modelname.split('/')[0] in model_A:
                ## CV
                avg = 'simple'
                method = 'CV'
                results = torch_find_scores(args.rounding, output_name, args.mode, avg, method, args.restrict_softmax)
            elif args.modelname.split('/')[0] in model_W:
                ## SEED
                method = 'SEED'
                results = torch_find_scores(args.rounding, output_name, args.mode, args.averaging, method, args.restrict_softmax)
    
    elif args.mode == 'check': ## to check confusion matrices
        print('---Check Predictions on TEST---')
        output_name = os.path.join('../torch_models', args.modelname)
        print(output_name)
        if args.modelname.split('/')[0] in model_A:
            avg = 'simple'
            method = 'CV'
            results = check_predictions(args.rounding, output_name, args.mode, avg, method, args.restrict_softmax)

    ## PDTB
    elif args.mode == 'pdtb':
        print('---Make Predictions on PDTB---')
        output_name = os.path.join('../torch_models', args.modelname)
        print(output_name)
        if args.modelname.split('/')[0] in model_A:
            avg = 'simple'
            method = 'while'
            results = torch_find_scores(args.rounding, output_name, args.mode, args.averaging, method, args.restrict_softmax)

if __name__ == '__main__':
    main()

