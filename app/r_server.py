import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from fastapi import FastAPI,Request,UploadFile 
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.concurrency import run_in_threadpool 
from app.query_gen import get_query
from app.query_retrival import make_retrival
import uvicorn
MODEL_VERSION='1.0.0'
# importing llm router function from your core logic script
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
# human readable 
@app.get('/')
def home():
    return {'message':'Telecom RCA analysis API'}
# for machine readable 
@app.get('/health')
def health_check():
    return {
        'status':'OK',
        'version': MODEL_VERSION
    }
# route for processing the query or the kpi file
@app.post('/api/process')
async def process_request(request:Request):
    # ---- SCENARIO A: KPI File Upload ----
    content_type=request.headers.get("content-type","")
    if "multipart/form-data" in content_type:
        form=await request.form()
        
        uploaded_file: UploadFile=form.get("file")
        if uploaded_file is None:
            return {"status":"error","message":"No file uploaded"}
        
        file_name = uploaded_file.filename
        file_bytes =await uploaded_file.read()
        
        # initializing default response string to prevent unbound variable crashes
        backend_result = "File uploaded, but no retrieval query was generated."
        # Import KPI-specific code only when a KPI file is actually uploaded.

        result, b = await run_in_threadpool(get_query,file_name, file_bytes)
        if b:
            # type 0 indicates it came from a kpi text block extraction
            backend_result =await run_in_threadpool(make_retrival,result, 0) 
            
        return {"status": "success","result": backend_result # standardized key to match streamlit query display
                }
        
    # ---- SCENARIO B: Plain Text Query ----
    elif "application/json" in content_type:
        json_data = await request.json()
        user_query = json_data.get('data')
        # making the retrieval for llm output,in synchronous 
        llm_output =await run_in_threadpool(make_retrival,user_query, 1)
        
        return {
            "status": "success",
            "result": llm_output
        }

    else:
        return {"status": "error", "message": "Invalid request format"}

if __name__ == '__main__':
    # set to 0.0.0.0 so your streamlit instance can access it over local network ports
    uvicorn.run(app, host="0.0.0.0", port=5001)