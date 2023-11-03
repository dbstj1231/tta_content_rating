import pandas as pd 
from jiwer import cer
import argparse


def main(config):
    import nemo.collections.asr as nemo_asr    
    bs = config.batch_size
    nw = config.num_wokers
    
    model_path ="./Conformer-CTC-BPE-8000-KOR.nemo"
    model = nemo_asr.models.ASRModel.restore_from(model_path)   
    df = pd.read_csv("./testset.csv")
    filelist= df.audio_filepath.to_list()
    model.eval()
    result = model.transcribe(filelist,batch_size=bs,num_workers=nw)    
    ref = df.text.to_list()    
    if config.write_result : 
        dict_list = [{"filepath":x,"reference":y,"prediction":z} for x,y,z in zip(filelist,ref,result)]
        result_path = "./result.csv"
        try :
            with open(result_path,'w',encoding='utf-8') as f:            
                df= pd.DataFrame(dict_list)
                df.prediction = df.prediction.str.replace("다","나")
                df.to_csv(f)
            print(f"result is saved {result_path}")
        except:
            raise
    df = pd.read_csv("./result.csv")
    df.prediction.fillna(" ",inplace=True)
    result = df.prediction.to_list()    
    ref = df.reference.to_list()
    _cer = cer(ref,result,return_dict=True) 
    print("".center(90,"-"))
    print("|"+f"Hits : {_cer['hits']}   Substitutions : {_cer['substitutions']}   Deletions : {_cer['deletions']}   Insertions : {_cer['insertions']}".center(88," ")+"|")
    print("|"+f"Accuracy : {(1-_cer['cer'])*100:.4f}% (CER is {_cer['cer']*100:.4f}%)".center(88," ")+"|")
    print("".center(90,"-"))
    
    
    
    

if __name__=="__main__":
    args = argparse.ArgumentParser()  
    args.add_argument('--batch_size', type=int, default=128,
                      help='batch size for inference(default : 128)')
    args.add_argument('--num_wokers', type=int, default=16,
                      help='num of workers for inference dataloader(default : 16)')   
    args.add_argument('--write_result', type=bool, default=True,
                      help='result write option(default : True)')   
    config = args.parse_args()
    main(config)