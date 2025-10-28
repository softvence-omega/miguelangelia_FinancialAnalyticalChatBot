from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.file_explore_endpoint import router as explore_router
# from app.routes.summary_and_report_generator_endpoint import router as summary_report_router
from app.routes.financial_bot_endpoint import router as financial_bot_router
from app.routes.general_bot_endpoint import router as general_bot_router
from app.routes.thread_creation_endpoint import router as session_router
from app.routes.summery_endpoint import router as summery_router
from app.routes.thread_deletion_endpoint import router as thread_deletion_router

app = FastAPI(title="Financial Analyst AI API")

# # ✅ 1️⃣ FIRST: Setup global error handlers (BEFORE adding routers)
# setup_global_error_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(explore_router, prefix='/ai')
# app.include_router(summary_report_router, prefix='/ai')
app.include_router(financial_bot_router, prefix='/ai')
app.include_router(general_bot_router, prefix='/ai')
app.include_router(session_router, prefix='/ai')
app.include_router(summery_router, prefix='/ai')

app.include_router(thread_deletion_router, prefix='/ai')

@app.get("/")
async def base_route():
    return {"message": "Welcome to the Financial Anantyc AI API"}

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

