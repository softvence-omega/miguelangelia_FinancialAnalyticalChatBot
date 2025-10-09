from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.file_explore_endpoint import router as explore_router
from app.routes.summary_and_report_generator_endpoint import router as summary_report_router

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

app.include_router(explore_router)
app.include_router(summary_report_router)

@app.get("/")
async def base_route():
    return {"message": "Welcome to the Financial Anantyc AI API"}

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

