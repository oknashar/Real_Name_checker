import time
from fastapi import FastAPI
import uvicorn
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse

from helpers import model_and_tokenizer
from shared_param import tokenizer_path ,model_path
from models import model
from test import predict_dl
# from models import LSTModel_1


app = FastAPI()

model ,tokenizer = model_and_tokenizer(model_path,tokenizer_path,model)

result = 0
x = 0


@app.get("/")
def get_name(name=''):
    response = RedirectResponse(url='/docs')
    if name == '':
        return response
    name = str.strip(name)
    if len(name.split()) <= 3 and len(name.split()) >= 1:
        start_time = time.time()
        res = predict_dl(model,tokenizer,name)
        fulltime =round(time.time() - start_time,2)
        if res > 0.35:


            final= {
                "الاسم":name,
                "الحاله":"الاسم صحيح",
                "واثق بنسبة":str(res),
                "الوقت المحتسب لتنفيذ العمليه":str(fulltime)
            }
            
        else:

            final= {
                "الاسم":name,
                "الحاله":"الاسم غير صحيح",
                "واثق بنسبة":str(1-res),
                "الوقت المحتسب لتنفيذ العمليه":str(fulltime)
            }
            
        res = jsonable_encoder(final)
        return str(res)
    else:
        return "Sorry!! You can only input 3 names "





if __name__ == "__main__":
    uvicorn.run(f"server:app", reload=True, host="0.0.0.0", port=8000)