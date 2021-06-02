import os
from tqdm import tqdm

from torch_train import torch_train
from torch_evaluate import torch_find_scores, torch_find_best, check_predictions

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m','--mode',choices=['train','dev','test','check'],help="dev:tune")
parser.add_argument('--data',choices=['amr','wiki','mix'],type=str)
parser.add_argument("-s","--sconj",help="True:with sconj,False:without sconj",action='store_true')
parser.add_argument("-n","--modelname",help="BERT-AMR/3_2e-05_16",type=str)
parser.add_argument("-d","--dirname",help="BERT-AMR",type=str,default=None)
parser.add_argument("-t","--target",help="BERT-WIKI/3_2e-05_64/0",type=str)

## For MIX models
parser.add_argument("--mix",help="1,2,3,4,5",type=int,default=1)
parser.add_argument("--mixid",help="1,2,3, ..., 25",type=int,default=1)

## Training Hyperparameter
parser.add_argument("-e","--epoch",help="epochs:3,5,10",type=int)
parser.add_argument("-l","--lr",help="learning_rate:2e-5,3e-5,5e-5",type=float)
parser.add_argument("-b","--batch",help="batch_size:16,32,64",type=int)

## Evaluation
parser.add_argument("-r","--rounding",help="rd: True",action='store_true')
parser.add_argument("-a","--averaging",help="simple,soft-vote",default='simple',type=str)
parser.add_argument("-o","--out_layer",help="Restriction: True, Vanilla: False",action='store_true',default=False)
args = parser.parse_args()

model_A = ['BERT-AMR', 'BERT-AMRs', 'BERT-MIX', 'BERT-MIXs', 'WIKI-AMR', 'WIKI-AMRs', 'WIKI-MIX']
model_W = ['BERT-WIKI', 'BERT-WIKIs']

rd = args.rounding
o = args.out_layer
r = args.mix

def main():
    if args.mode == 'train':
        if args.data == 'amr':
            if args.target:
                print('WIKI-AMR') ## BERT-WIKI-AMR
                model_to_tune = str(args.target)
                model_name = str(args.epoch)+'_'+str(args.lr)+'_'+str(args.batch)
                for cv_num in tqdm(range(1,6)):
                    torch_train(model_name, args.data, args.sconj, args.mix, args.mixid, args.epoch, args.lr, args.batch, cv_num, model_to_tune)

            else:
                print('BERT-AMR')
                model_name = str(args.epoch)+'_'+str(args.lr)+'_'+str(args.batch)
                for cv_num in tqdm(range(1,6)):
                    torch_train(model_name, args.data, args.sconj, args.mix, args.mixid, args.epoch, args.lr, args.batch, cv_num)

        elif args.data == 'mix':
            if args.target:
                print('WIKI-MIX') ## BERT-WIKI-MIX
                model_to_tune = str(args.target)
                model_name = str(args.epoch)+'_'+str(args.lr)+'_'+str(args.batch)
                for cv_num in tqdm(range(1,6)):
                    torch_train(model_name, args.data, args.sconj, args.mix, args.mixid, args.epoch, args.lr, args.batch, cv_num, model_to_tune)

            else:
                print('BERT-MIX')
                model_name = str(args.epoch)+'_'+str(args.lr)+'_'+str(args.batch)
                for cv_num in tqdm(range(1,6)):
                    torch_train(model_name, args.data, args.sconj, args.mix, args.mixid, args.epoch, args.lr, args.batch, cv_num)

        elif args.data == 'wiki':
            print('---Train on WIKI---')
            print('BERT-WIKI')
            model_name = str(args.epoch)+'_'+str(args.lr)+'_'+str(args.batch)
            for seed_num in tqdm(range(5)):
                torch_train(model_name, args.data, args.sconj, args.mix, args.mixid, args.epoch, args.lr, args.batch, seed_num)
        print('---Done Training---')

    elif args.mode == 'dev':
        #args.modelname: BERT-AMR/3_3e-05_16
        if args.modelname:
            print('---Evaluate on DEV---')
            if args.modelname.split('/')[0] in model_A:
                output_name = os.path.join('torch_models',args.modelname)
                method = 'CV'
                avg = 'simple'
                micro = torch_find_scores(rd, output_name, args.mode, avg, method)
                print('micro-f1: '+str(micro))
            elif args.modelname.split('/')[0] in model_W:
                output_name = os.path.join('torch_models',args.modelname)
                method = 'SEED'
                micro = torch_find_scores(rd, output_name, args.mode, args.averaging, method)
                print('micro-f1: '+str(micro))
        
        elif args.dirname:
            #args.dirname: BERT-AMR
            print('---Find Best Model on DEV---')
            if args.dirname in model_A:
                results = torch_find_best(rd, args.dirname, avg='simple', method='CV')
                print('---Score Ranking: '+str(args.dirname)+'---')
                for model, score in results:
                    print(model+': '+str(score))
            elif args.dirname in model_W:
                results = torch_find_best(rd, args.dirname, args.averaging, method='SEED')
                print('---Score Ranking: '+str(args.dirname)+'---')
                for model, score in results:
                    print(model+': '+str(score))

    elif args.mode == 'test':
        print('---Evaluate on TEST---')
        if args.dirname == 'BERT-MIX':
            ## SEED & CV
            output_name = os.path.join('torch_models', args.dirname)
            avg = 'simple'
            method = 'MIX_CV'
            results = torch_find_scores(rd, output_name, args.mode, avg, method, o, r)

        else:
            output_name = os.path.join('torch_models', args.modelname)
            print(output_name)
            if args.modelname.split('/')[0] in model_A:
                ## CV
                avg = 'simple'
                method = 'CV'
                results = torch_find_scores(rd, output_name, args.mode, avg, method, o)
            elif args.modelname.split('/')[0] in model_W:
                ## SEED
                method = 'SEED'
                results = torch_find_scores(rd, output_name, args.mode, args.averaging, method, o)
    
    elif args.mode == 'check': ## to check confusion matrices
        print('---Check Predictions on TEST---')
        output_name = os.path.join('torch_models', args.modelname)
        print(output_name)
        if args.modelname.split('/')[0] in model_A:
            avg = 'simple'
            method = 'CV'
            results = check_predictions(rd, output_name, args.mode, avg, method, o)

if __name__ == '__main__':
    main()
