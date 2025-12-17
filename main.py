"""
main.py - FastAPI 主入口

职责：
- HTTP 服务启动
- 路由定义
- 中间件配置
- 集成 RequestManager 进行并发控制
"""

import json
import os
import time
import logging
import asyncio
import uuid
import queue
import threading
import re
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from collections import deque
import threading

from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, FileResponse, Response
from pydantic import BaseModel, Field

from browser_core import get_browser, BrowserConnectionError
from config_engine import config_engine, ConfigConstants
from request_manager import request_manager, RequestContext, RequestStatus, watch_client_disconnect
from data_models import (
    ChatCompletionRequest, 
    HealthCheckResult,
    ModelsResponse,
    ModelInfo,
    SiteConfig
)


# ================= 环境变量配置 =================

class AppConfig:
    """应用配置"""
    HOST = os.getenv("APP_HOST", "127.0.0.1")
    PORT = int(os.getenv("APP_PORT", "8199"))
    DEBUG = os.getenv("APP_DEBUG", "false").lower() == "true"
    
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    CORS_ENABLED = os.getenv("CORS_ENABLED", "true").lower() == "true"
    
    AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"
    AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
    
    BROWSER_PORT = int(os.getenv("BROWSER_PORT", "9222"))
    
    DASHBOARD_ENABLED = os.getenv("DASHBOARD_ENABLED", "true").lower() == "true"
    DASHBOARD_FILE = os.getenv("DASHBOARD_FILE", "dashboard.html")
    DASHBOARD_JS_FILE = os.getenv("DASHBOARD_JS_FILE", "dashboard.js")
    
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# ================= 日志配置 =================

logging.basicConfig(
    level=getattr(logging, AppConfig.LOG_LEVEL),
    format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger('main')


# ================= 日志收集器 =================

class LogCollector:
    """收集日志用于前端展示"""
    
    def __init__(self, max_logs=500):
        self.logs = deque(maxlen=max_logs)
        self.lock = threading.Lock()
    
    def add(self, level: str, message: str):
        with self.lock:
            self.logs.append({
                "timestamp": time.time(),
                "level": level,
                "message": message
            })
    
    def get_recent(self, since: float = 0):
        with self.lock:
            recent = [log for log in self.logs if log["timestamp"] > since]
            return list(recent)
    
    def clear(self):
        with self.lock:
            self.logs.clear()


log_collector = LogCollector()


class WebLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            log_collector.add(record.levelname, msg)
        except Exception:
            self.handleError(record)


web_handler = WebLogHandler()
web_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(web_handler)


# ================= Lifespan =================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Universal Web-to-API 服务启动中...")
    logger.info(f"监听地址: http://{AppConfig.HOST}:{AppConfig.PORT}")
    logger.info(f"调试模式: {AppConfig.DEBUG}")
    logger.info(f"认证: {'启用' if AppConfig.AUTH_ENABLED else '禁用'}")
    logger.info(f"浏览器端口: {AppConfig.BROWSER_PORT}")
    logger.info("=" * 60)
    
    # 浏览器健康检查（延迟初始化，不阻塞启动）
    try:
        browser = get_browser(auto_connect=False)
        health = browser.health_check()
        if health["connected"]:
            logger.info(f"✅ 浏览器已连接: {health['tab_url']}")
        else:
            logger.warning(f"⚠️ 浏览器未连接: {health.get('error', '未知')}")
    except Exception as e:
        logger.warning(f"⚠️ 浏览器检查跳过: {e}")
    
    logger.info("")
    logger.info("🚀 服务已就绪！")
    if AppConfig.DASHBOARD_ENABLED:
        logger.info(f"   Dashboard: http://{AppConfig.HOST}:{AppConfig.PORT}/dashboard")
    logger.info(f"   健康检查: http://{AppConfig.HOST}:{AppConfig.PORT}/health")
    logger.info("")

    yield

    logger.info("服务正在关闭...")
    
    try:
        browser = get_browser(auto_connect=False)
        browser.close()
    except Exception as e:
        logger.debug(f"关闭浏览器: {e}")
    
    logger.info("👋 服务已停止")


# ================= FastAPI 应用 =================

app = FastAPI(
    title="Universal Web-to-API",
    description="将任意 AI Web 界面转换为 OpenAI 兼容 API",
    version="2.0.0",
    docs_url="/docs" if AppConfig.DEBUG else None,
    redoc_url="/redoc" if AppConfig.DEBUG else None,
    lifespan=lifespan
)


# ================= CORS =================

if AppConfig.CORS_ENABLED:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=AppConfig.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ================= 请求模型 =================

class ChatRequest(BaseModel):
    model: str = Field(default="gpt-3.5-turbo")
    messages: list = Field(...)
    stream: Optional[bool] = Field(default=True)
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=1)


class ConfigUpdateRequest(BaseModel):
    config: Dict[str, Any] = Field(...)


# ================= 认证 =================

async def verify_auth(authorization: Optional[str] = Header(None)) -> bool:
    if not AppConfig.AUTH_ENABLED:
        return True
    
    if not AppConfig.AUTH_TOKEN:
        raise HTTPException(status_code=500, detail="服务配置错误")
    
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="未提供认证令牌",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = authorization.replace("Bearer ", "").strip()
    
    if token != AppConfig.AUTH_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="认证令牌无效",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return True


# ================= 核心 API =================

@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatRequest,
    authenticated: bool = Depends(verify_auth)
):
    """
    OpenAI 兼容的聊天补全接口
    
    集成 RequestManager 进行：
    1. 请求排队（FIFO）
    2. 并发控制
    3. 客户端断开检测
    4. 优雅取消
    """
    
    # 创建请求上下文
    ctx = request_manager.create_request()
    logger.info(f"请求 [{ctx.request_id}] 开始...")
    
    if body.stream:
        return StreamingResponse(
            _stream_with_lifecycle(request, body, ctx),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        return await _non_stream_with_lifecycle(request, body, ctx)


async def _stream_with_lifecycle(
    request: Request,
    body: ChatRequest,
    ctx: RequestContext
):
    """
    流式响应 + 完整生命周期管理
    
    关键修复：
    1. 生成器在独立线程执行（不阻塞事件循环）
    2. 主协程定期检查连接状态
    3. 通过队列传递数据
    """
    
    disconnect_task = None
    worker_thread = None
    acquired = False
    
    try:
        # ========== 1. 获取锁 ==========
        try:
            acquired = await request_manager.acquire(ctx, timeout=60.0)
        except asyncio.CancelledError:
            logger.info(f"请求 [{ctx.request_id}] 在排队时被取消")
            yield _pack_error("请求已取消", "cancelled")
            yield _pack_done()
            return
        
        if not acquired:
            reason = ctx.cancel_reason or "获取锁失败"
            logger.warning(f"请求 [{ctx.request_id}] {reason}")
            yield _pack_error(f"服务繁忙: {reason}", "busy")
            yield _pack_done()
            return
        
        # ========== 2. 启动断开检测 ==========
        disconnect_task = asyncio.create_task(
            watch_client_disconnect(request, ctx, check_interval=0.3)
        )
        
        # ========== 3. 设置浏览器停止检查器 ==========
        browser = get_browser(auto_connect=False)
        browser.set_stop_checker(ctx.should_stop)
        
        # ========== 4. 创建队列 + 工作线程 ==========
        chunk_queue = queue.Queue(maxsize=100)  # 限制队列大小防止内存溢出
        
        def worker():
            """在独立线程中执行同步生成器"""
            try:
                logger.info(f"请求 [{ctx.request_id}] 工作线程启动")
                gen = browser.execute_workflow(body.messages, stream=True)
                
                for chunk in gen:
                    # 检查取消标志
                    if ctx.should_stop():
                        logger.info(f"请求 [{ctx.request_id}] 工作线程检测到取消")
                        break
                    
                    # 放入队列
                    chunk_queue.put(chunk)
                
            except Exception as e:
                logger.error(f"请求 [{ctx.request_id}] 工作线程异常: {e}")
                chunk_queue.put(("ERROR", str(e)))
            
            finally:
                # 发送结束标记
                chunk_queue.put(None)
                logger.debug(f"请求 [{ctx.request_id}] 工作线程结束")
        
        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()
        
        logger.info(f"请求 [{ctx.request_id}] 正在执行工作流...")
        
        # ========== 5. 从队列读取并发送 ==========
        while True:
            # 非阻塞地检查连接状态
            if await request.is_disconnected():
                logger.info(f"请求 [{ctx.request_id}] 检测到客户端断开")
                ctx.request_cancel("client_disconnected")
                break
            
            # 带超时地从队列获取
            try:
                chunk = await asyncio.to_thread(chunk_queue.get, timeout=0.5)
            except queue.Empty:
                # 超时，继续循环检查连接
                continue
            
            # 结束标记
            if chunk is None:
                logger.debug(f"请求 [{ctx.request_id}] 收到结束标记")
                break
            
            # 错误标记
            if isinstance(chunk, tuple) and chunk[0] == "ERROR":
                logger.error(f"请求 [{ctx.request_id}] 收到错误: {chunk[1]}")
                ctx.mark_failed(chunk[1])
                yield _pack_error(f"执行错误: {chunk[1]}", "internal_error")
                break
            
            # 正常数据
            yield chunk
            
            # 让出控制权，允许其他协程运行
            await asyncio.sleep(0)
        
        # 如果没有被取消且没有错误，标记完成
        if not ctx.should_stop() and ctx.status == RequestStatus.RUNNING:
            ctx.mark_completed()
    
    except asyncio.CancelledError:
        logger.info(f"请求 [{ctx.request_id}] 协程被取消")
        ctx.request_cancel("coroutine_cancelled")
        raise
    
    except Exception as e:
        logger.error(f"请求 [{ctx.request_id}] 异常: {e}", exc_info=True)
        ctx.mark_failed(str(e))
        yield _pack_error(f"执行错误: {str(e)}", "internal_error")
    
    finally:
        # ========== 6. 清理 ==========
        
        # 如果工作线程还在运行，设置取消标志
        if worker_thread and worker_thread.is_alive():
            ctx.request_cancel("cleanup")
            
            # 等待线程结束（最多 2 秒）
            worker_thread.join(timeout=2.0)
            
            if worker_thread.is_alive():
                logger.warning(f"请求 [{ctx.request_id}] 工作线程未能及时结束")
        
        # 清空队列
        try:
            while not chunk_queue.empty():
                chunk_queue.get_nowait()
        except:
            pass
        
        # 取消断开检测
        if disconnect_task:
            disconnect_task.cancel()
            try:
                await disconnect_task
            except asyncio.CancelledError:
                pass
        
        # 释放锁
        if acquired:
            request_manager.release(ctx, success=(ctx.status == RequestStatus.COMPLETED))
        
        logger.info(f"请求 [{ctx.request_id}] 结束 (状态: {ctx.status.value})")


async def _non_stream_with_lifecycle(
    request: Request,
    body: ChatRequest,
    ctx: RequestContext
) -> JSONResponse:
    """非流式响应 + 生命周期管理"""
    
    collected_content = []
    error_data = None
    
    async for chunk in _stream_with_lifecycle(request, body, ctx):
        if isinstance(chunk, str):
            if chunk.startswith("data: [DONE]"):
                continue
            
            if chunk.startswith("data: "):
                try:
                    data_str = chunk[6:].strip()
                    if not data_str:
                        continue
                    data = json.loads(data_str)
                    
                    if "error" in data:
                        error_data = data
                        break
                    
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            collected_content.append(content)
                except json.JSONDecodeError:
                    continue
    
    if error_data:
        return JSONResponse(content=error_data, status_code=500)
    
    full_content = "".join(collected_content)
    response = {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": full_content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
    
    return JSONResponse(content=response)


def _pack_error(message: str, code: str = "error") -> str:
    """打包 SSE 错误"""
    data = {
        "error": {
            "message": message,
            "type": "execution_error",
            "code": code
        }
    }
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _pack_done() -> str:
    """打包 SSE 结束标记"""
    return "data: [DONE]\n\n"


# ================= 模型列表 =================

@app.get("/v1/models")
async def list_models(authenticated: bool = Depends(verify_auth)):
    return {
        "object": "list",
        "data": [
            {
                "id": "web-browser",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "universal-web-api"
            }
        ]
    }


# ================= 健康检查 =================

@app.get("/health")
async def health_check():
    try:
        browser = get_browser(auto_connect=False)
        browser_health = browser.health_check()
    except Exception as e:
        browser_health = {"connected": False, "error": str(e)}
    
    # 请求管理器状态
    rm_status = request_manager.get_status()
    
    response = {
        "service": "healthy",
        "version": "2.0.0",
        "browser": browser_health,
        "request_manager": rm_status,
        "config": {
            "sites_loaded": len(config_engine.sites),
            "auth_enabled": AppConfig.AUTH_ENABLED
        },
        "timestamp": int(time.time())
    }
    
    status_code = 200 if browser_health.get("connected") else 503
    return JSONResponse(content=response, status_code=status_code)


# ================= 配置管理 API =================

@app.get("/api/config")
async def get_config(authenticated: bool = Depends(verify_auth)):
    try:
        if os.path.exists(ConfigConstants.CONFIG_FILE):
            with open(ConfigConstants.CONFIG_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        return {}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail="配置文件格式错误")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/config")
async def save_config(
    request: ConfigUpdateRequest,
    authenticated: bool = Depends(verify_auth)
):
    try:
        with open(ConfigConstants.CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(request.config, f, indent=2, ensure_ascii=False)
        
        config_engine.reload_config()
        
        return {
            "status": "success",
            "message": "配置已保存",
            "sites_count": len(request.config)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/config/{domain}")
async def delete_site_config(
    domain: str,
    authenticated: bool = Depends(verify_auth)
):
    success = config_engine.delete_site_config(domain)
    
    if success:
        return {"status": "success", "message": f"已删除: {domain}"}
    else:
        raise HTTPException(status_code=404, detail=f"配置不存在: {domain}")


@app.get("/api/config/{domain}")
async def get_site_config(
    domain: str,
    authenticated: bool = Depends(verify_auth)
):
    if domain in config_engine.sites:
        return config_engine.sites[domain]
    else:
        raise HTTPException(status_code=404, detail=f"配置不存在: {domain}")



# ================= 系统设置 API =================

@app.get("/api/settings/env")
async def get_env_config(authenticated: bool = Depends(verify_auth)):
    """
    读取 .env 文件配置
    """
    try:
        env_path = Path(".env")
        config = {}
        
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # 跳过注释和空行
                    if not line or line.startswith('#'):
                        continue
                    
                    # 解析 KEY=VALUE
                    if '=' not in line:
                        continue
                    
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 类型转换
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    elif re.match(r'^\d+\.\d+$', value):
                        value = float(value)
                    
                    config[key] = value
        
        return {"config": config}
    
    except Exception as e:
        logger.error(f"读取环境配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"读取失败: {str(e)}")


@app.post("/api/settings/env")
async def save_env_config(
    request: Request,
    authenticated: bool = Depends(verify_auth)
):
    """
    保存 .env 配置（保留注释结构）
    """
    try:
        data = await request.json()
        new_config = data.get("config", {})
        
        env_path = Path(".env")
        lines = []
        
        # 读取现有文件
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        
        new_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # 保留注释和空行
            if not stripped or stripped.startswith('#'):
                new_lines.append(line)
                continue
            
            # 处理配置行
            if '=' in stripped:
                key = stripped.split('=', 1)[0].strip()
                
                if key in new_config:
                    # 替换值
                    value = new_config[key]
                    
                    # 类型转换
                    if isinstance(value, bool):
                        value = 'true' if value else 'false'
                    elif isinstance(value, (int, float)):
                        value = str(value)
                    
                    new_lines.append(f"{key}={value}\n")
                else:
                    # 保留原行
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        # 写回文件
        with open(env_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        logger.info(f"环境配置已保存: {len(new_config)} 项")
        
        return {
            "status": "success",
            "message": "环境配置已保存（部分配置需重启生效）",
            "updated_count": len(new_config)
        }
    
    except Exception as e:
        logger.error(f"保存环境配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")


@app.get("/api/settings/browser-constants")
async def get_browser_constants(authenticated: bool = Depends(verify_auth)):
    """
    读取浏览器常量配置
    """
    try:
        config_path = Path("browser_config.json")
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            # 返回默认值
            from browser_core import BrowserConstants
            
            # 如果 BrowserConstants 还没有 get_defaults 方法，返回硬编码的默认值
            config = {
                'DEFAULT_PORT': 9222,
                'CONNECTION_TIMEOUT': 10,
                'STEALTH_DELAY_MIN': 0.1,
                'STEALTH_DELAY_MAX': 0.3,
                'ACTION_DELAY_MIN': 0.15,
                'ACTION_DELAY_MAX': 0.3,
                'DEFAULT_ELEMENT_TIMEOUT': 3,
                'FALLBACK_ELEMENT_TIMEOUT': 1,
                'ELEMENT_CACHE_MAX_AGE': 5.0,
                'STREAM_CHECK_INTERVAL_MIN': 0.1,
                'STREAM_CHECK_INTERVAL_MAX': 1.0,
                'STREAM_CHECK_INTERVAL_DEFAULT': 0.3,
                'STREAM_SILENCE_THRESHOLD': 8.0,
                'STREAM_MAX_TIMEOUT': 600,
                'STREAM_INITIAL_WAIT': 180,
                'STREAM_RERENDER_WAIT': 0.5,
                'STREAM_CONTENT_SHRINK_TOLERANCE': 3,
                'STREAM_MIN_VALID_LENGTH': 10,
                'STREAM_STABLE_COUNT_THRESHOLD': 8,
                'STREAM_SILENCE_THRESHOLD_FALLBACK': 12,
                'MAX_MESSAGE_LENGTH': 100000,
                'MAX_MESSAGES_COUNT': 100,
                'STREAM_INITIAL_ELEMENT_WAIT': 10,
                'STREAM_MAX_ABNORMAL_COUNT': 5,
                'STREAM_MAX_ELEMENT_MISSING': 10,
                'STREAM_CONTENT_SHRINK_THRESHOLD': 0.3,
            }
        
        return {"config": config}
    
    except Exception as e:
        logger.error(f"读取浏览器常量失败: {e}")
        raise HTTPException(status_code=500, detail=f"读取失败: {str(e)}")


@app.post("/api/settings/browser-constants")
async def save_browser_constants(
    request: Request,
    authenticated: bool = Depends(verify_auth)
):
    """
    保存浏览器常量配置
    """
    try:
        data = await request.json()
        config = data.get("config", {})
        
        config_path = Path("browser_config.json")
        
        # 保存到文件
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 尝试热重载
        try:
            from browser_core import BrowserConstants
            if hasattr(BrowserConstants, 'reload'):
                BrowserConstants.reload()
                logger.info("浏览器常量已热重载")
            else:
                logger.warning("BrowserConstants 不支持热重载，需重启服务")
        except Exception as reload_error:
            logger.warning(f"热重载失败: {reload_error}")
        
        logger.info(f"浏览器常量已保存: {len(config)} 项")
        
        return {
            "status": "success",
            "message": "浏览器常量已保存",
            "updated_count": len(config)
        }
    
    except Exception as e:
        logger.error(f"保存浏览器常量失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")
# ================= 日志 API =================

@app.get("/api/logs")
async def get_logs(since: float = 0, authenticated: bool = Depends(verify_auth)):
    logs = log_collector.get_recent(since)
    return {"logs": logs, "timestamp": time.time()}


@app.delete("/api/logs")
async def clear_logs(authenticated: bool = Depends(verify_auth)):
    log_collector.clear()
    return {"status": "success"}


# ================= 调试 API =================

@app.post("/api/debug/test-selector")
async def test_selector(
    request: Request,
    authenticated: bool = Depends(verify_auth)
):
    if not AppConfig.DEBUG:
        raise HTTPException(status_code=403, detail="调试功能未启用")
    
    try:
        data = await request.json()
        selector = data.get("selector", "")
        timeout = data.get("timeout", 2)
        highlight = data.get("highlight", False)
        
        if not selector:
            raise HTTPException(status_code=400, detail="缺少 selector")
        
        browser = get_browser()
        tab = browser.get_active_tab()
        elements = tab.eles(selector, timeout=timeout)
        
        if not elements:
            return {"success": False, "count": 0, "message": "元素未找到"}
        
        if not isinstance(elements, list):
            elements = [elements]
        
        result = {
            "success": True,
            "count": len(elements),
            "elements": []
        }
        
        for idx, ele in enumerate(elements):
            result["elements"].append({
                "index": idx,
                "tag": ele.tag,
                "text": ele.text[:100] if ele.text else ""
            })
            
            if highlight:
                try:
                    tab.run_js(f"""
                        (function() {{
                            const elements = document.querySelectorAll('{selector}');
                            if (elements[{idx}]) {{
                                const el = elements[{idx}];
                                el.style.outline = '3px solid red';
                                el.style.outlineOffset = '2px';
                                setTimeout(() => {{
                                    el.style.outline = '';
                                    el.style.outlineOffset = '';
                                }}, 5000);
                            }}
                        }})();
                    """)
                except Exception:
                    pass
        
        return result
    
    except Exception as e:
        return {"success": False, "count": 0, "message": str(e)}


@app.get("/api/debug/request-status")
async def request_status(authenticated: bool = Depends(verify_auth)):
    """查看请求管理器状态"""
    return request_manager.get_status()


@app.post("/api/debug/force-release")
async def force_release(authenticated: bool = Depends(verify_auth)):
    """强制释放锁（紧急情况）"""
    if not AppConfig.DEBUG:
        raise HTTPException(status_code=403, detail="调试功能未启用")
    
    was_locked = request_manager.is_locked()
    released = request_manager.force_release()
    is_now_locked = request_manager.is_locked()
    
    logger.warning(f"手动解锁: was={was_locked}, released={released}, now={is_now_locked}")
    
    return {
        "was_locked": was_locked,
        "released": released,
        "is_now_locked": is_now_locked
    }


@app.post("/api/debug/cancel-current")
async def cancel_current(authenticated: bool = Depends(verify_auth)):
    """取消当前正在执行的请求"""
    current_id = request_manager.get_current_request_id()
    
    if not current_id:
        return {"cancelled": False, "message": "没有正在执行的请求"}
    
    success = request_manager.cancel_current("manual_cancel")
    
    return {
        "cancelled": success,
        "request_id": current_id
    }


# ================= Dashboard =================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    if not AppConfig.DASHBOARD_ENABLED:
        raise HTTPException(status_code=403, detail="Dashboard 未启用")
    
    try:
        dashboard_path = Path(AppConfig.DASHBOARD_FILE)
        
        if dashboard_path.exists():
            content = dashboard_path.read_text(encoding="utf-8")
            return HTMLResponse(content=content)
        else:
            return HTMLResponse(
                content="<h1>Dashboard 文件未找到</h1>",
                status_code=404
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard.js")
async def dashboard_js():
    js_path = Path(AppConfig.DASHBOARD_JS_FILE)
    if js_path.exists():
        return FileResponse(js_path, media_type="application/javascript")
    else:
        raise HTTPException(status_code=404)


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/")
async def root():
    return {
        "service": "Universal Web-to-API",
        "version": "2.0.0",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health",
            "dashboard": "/dashboard"
        }
    }


# ================= 异常处理 =================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"error": {"message": "接口不存在", "path": str(request.url.path)}}
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    logger.error(f"内部错误: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "服务器内部错误"}}
    )


# ================= 主入口 =================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("环境变量配置（可选）:")
    print("  APP_HOST=0.0.0.0          # 监听地址")
    print("  APP_PORT=8199             # 监听端口")
    print("  APP_DEBUG=true            # 调试模式")
    print("  AUTH_ENABLED=true         # 启用认证")
    print("  AUTH_TOKEN=your-secret    # 认证令牌")
    print("  BROWSER_PORT=9222         # 浏览器端口")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        app,
        host=AppConfig.HOST,
        port=AppConfig.PORT,
        log_level=AppConfig.LOG_LEVEL.lower(),
        access_log=False
    )
