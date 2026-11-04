# main.py
import uvicorn
from fastapi import FastAPI

import routers
from exception import register_exception_handlers

app = FastAPI()

register_exception_handlers(app)

app.include_router(routers.router)

if __name__ == "__main__":
    # 推荐传入字符串形式 "main:app"，可保证热重载正常生效
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发环境开启
        # workers=4   # 生产环境开启多进程，与 reload 不可同时生效
    )
