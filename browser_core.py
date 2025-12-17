"""
browser_core.py - 浏览器自动化核心模块

职责：
- 浏览器连接管理
- 工作流执行
- 元素查找与操作

重构说明（v2）：
- 移除内部锁管理，由 RequestManager 统一控制
- 接收外部的取消信号检查器
- 增强取消响应速度
"""

import os
import time
import json
import random
import logging
import threading
import hashlib
import uuid
from pathlib import Path
from typing import Generator, Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from DrissionPage import ChromiumPage


# ================= 常量配置 =================

class BrowserConstants:
    """浏览器相关常量"""
    
    # ===== 新增：配置缓存 =====
    _config = None
    _config_file = Path("browser_config.json")
    
    # ===== 新增：默认值字典 =====
    _DEFAULTS = {
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
    # 连接配置
    DEFAULT_PORT = 9222
    CONNECTION_TIMEOUT = 10
    
    # 延迟配置
    STEALTH_DELAY_MIN = 0.1
    STEALTH_DELAY_MAX = 0.3
    ACTION_DELAY_MIN = 0.15
    ACTION_DELAY_MAX = 0.3
    
    # 元素查找
    DEFAULT_ELEMENT_TIMEOUT = 3
    FALLBACK_ELEMENT_TIMEOUT = 1
    ELEMENT_CACHE_MAX_AGE = 5.0
    
    # 流式监控（优化：更短间隔，更快响应取消）
    STREAM_CHECK_INTERVAL_MIN = 0.1
    STREAM_CHECK_INTERVAL_MAX = 1.0          # ✅ 从 0.5 增加到 1.0 秒
    STREAM_CHECK_INTERVAL_DEFAULT = 0.3      # ✅ 从 0.2 增加到 0.3 秒
    
    STREAM_SILENCE_THRESHOLD = 8.0           # ✅ 从 3.5 增加到 8 秒
    # 关键：慢速 AI 两次更新可能间隔 5-10 秒
    
    STREAM_MAX_TIMEOUT = 600
    STREAM_INITIAL_WAIT = 180                # ✅ 从 120 增加到 180 秒（3分钟）
    
    # 流式监控增强配置
    STREAM_RERENDER_WAIT = 0.5
    STREAM_CONTENT_SHRINK_TOLERANCE = 3
    STREAM_MIN_VALID_LENGTH = 10
    
    STREAM_STABLE_COUNT_THRESHOLD = 8        # ✅ 从 4 增加到 8 次
    # 需要连续 8 次检查不变才判定稳定
    
    STREAM_SILENCE_THRESHOLD_FALLBACK = 12   # ✅ 从 6 增加到 12 秒
    
    # 输入验证
    MAX_MESSAGE_LENGTH = 100000
    MAX_MESSAGES_COUNT = 100

    # 异常检测配置
    STREAM_INITIAL_ELEMENT_WAIT = 10
    STREAM_MAX_ABNORMAL_COUNT = 5
    STREAM_MAX_ELEMENT_MISSING = 10
    STREAM_CONTENT_SHRINK_THRESHOLD = 0.3

    @classmethod
    def _load_config(cls):
        """从文件加载配置"""
        if cls._config_file.exists():
            try:
                import json
                with open(cls._config_file, 'r', encoding='utf-8') as f:
                    cls._config = json.load(f)
                return
            except Exception:
                pass
        
        # 加载失败或文件不存在，使用默认值
        cls._config = cls._DEFAULTS.copy()
    
    @classmethod
    def get(cls, key: str):
        """获取配置值（支持动态加载）"""
        if cls._config is None:
            cls._load_config()
        
        return cls._config.get(key, cls._DEFAULTS.get(key))
    
    @classmethod
    def get_defaults(cls):
        """获取所有默认值"""
        return cls._DEFAULTS.copy()
    
    @classmethod
    def reload(cls):
        """重新加载配置（热重载）"""
        cls._config = None
        cls._load_config()
# ================= 安全日志配置 =================

class SecureLogger:
    """安全日志封装器"""
    
    LOG_SENSITIVE = os.environ.get('BROWSER_LOG_SENSITIVE', 'false').lower() == 'true'
    
    def __init__(self, name: str, level: int = logging.INFO):
        self._logger = self._setup_logger(name, level)
    
    def _setup_logger(self, name: str, level: int) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(level)
        return logger
    
    def info(self, msg: str, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)
    
    def info_sensitive(self, msg: str, content: str = None,
                       max_preview: int = 50, *args, **kwargs):
        if content is None:
            self._logger.info(msg, *args, **kwargs)
            return
        
        if self.LOG_SENSITIVE:
            preview = content[:max_preview] + "..." if len(content) > max_preview else content
            self._logger.info(f"{msg} | preview='{preview}'", *args, **kwargs)
        else:
            self._logger.info(f"{msg} | len={len(content)}", *args, **kwargs)


logger = SecureLogger('browser')


# ================= 异常定义 =================

class BrowserError(Exception):
    """浏览器相关错误基类"""
    pass


class BrowserConnectionError(BrowserError):
    """浏览器连接错误"""
    pass


class ElementNotFoundError(BrowserError):
    """元素未找到错误"""
    pass


class WorkflowError(BrowserError):
    """工作流执行错误"""
    pass


class WorkflowCancelledError(WorkflowError):
    """工作流被取消"""
    pass


class ConfigurationError(BrowserError):
    """配置错误"""
    pass


# ================= SSE 格式化器 =================

class SSEFormatter:
    """SSE 响应格式化器"""
    
    _sequence = 0
    _sequence_lock = threading.Lock()
    
    @classmethod
    def _generate_id(cls) -> str:
        timestamp = int(time.time() * 1000)
        with cls._sequence_lock:
            cls._sequence += 1
            seq = cls._sequence
        short_uuid = uuid.uuid4().hex[:6]
        return f"chatcmpl-{timestamp}-{seq}-{short_uuid}"
    
    @classmethod
    def pack_chunk(cls, content: str, model: str = "web-browser") -> str:
        data = {
            "id": cls._generate_id(),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None
            }]
        }
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    
    @classmethod
    def pack_finish(cls, model: str = "web-browser") -> str:
        data = {
            "id": cls._generate_id(),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\ndata: [DONE]\n\n"
    
    @staticmethod
    def pack_error(message: str, error_type: str = "execution_error",
                   code: str = "workflow_failed") -> str:
        data = {
            "error": {
                "message": message,
                "type": error_type,
                "code": code
            }
        }
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    
    @staticmethod
    def pack_error_json(message: str, error_type: str = "execution_error",
                        code: str = "workflow_failed") -> Dict:
        return {
            "error": {
                "message": message,
                "type": error_type,
                "code": code
            }
        }
    
    @staticmethod
    def pack_non_stream(content: str, model: str = "web-browser") -> Dict:
        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }


# ================= 消息验证器 =================

class MessageValidator:
    """消息验证器"""
    
    VALID_ROLES = {'user', 'assistant', 'system'}
    
    @classmethod
    def validate(cls, messages: Any) -> tuple:
        if messages is None:
            return False, "messages 不能为空", None
        
        if not isinstance(messages, list):
            return False, f"messages 应该是列表", None
        
        if len(messages) == 0:
            return False, "messages 不能为空列表", None
        
        if len(messages) > BrowserConstants.MAX_MESSAGES_COUNT:
            return False, f"消息数量超过限制", None
        
        sanitized = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                return False, f"messages[{i}] 不是字典类型", None
            
            role = msg.get('role', 'user')
            if role not in cls.VALID_ROLES:
                role = 'user'
            
            content = msg.get('content', '')
            if not isinstance(content, str):
                content = str(content) if content is not None else ''
            
            if len(content) > BrowserConstants.MAX_MESSAGE_LENGTH:
                return False, f"messages[{i}].content 超过长度限制", None
            
            sanitized.append({'role': role, 'content': content})
        
        return True, None, sanitized


# ================= 缓存元素 =================

@dataclass
class CachedElement:
    element: Any
    selector: str
    cached_at: float
    content_hash: str
    
    def is_stale(self, max_age: float = None) -> bool:
        max_age = max_age or BrowserConstants.ELEMENT_CACHE_MAX_AGE
        return time.time() - self.cached_at > max_age


# ================= 元素查找器 =================

class ElementFinder:
    """元素查找器"""
    
    FALLBACK_SELECTORS: Dict[str, List[str]] = {
        "input_box": [
            'tag:textarea',
            'css:textarea',
            'css:textarea[name="message"]',
            'css:textarea[placeholder]',
            'tag:div@@contenteditable=true',
            'css:[contenteditable="true"]',
        ],
        "send_btn": [
            'css:button[type="submit"]',
            'tag:button@@type=submit',
            'css:form button[type="submit"]',
            'css:[role="button"][type="submit"]',
        ],
        "result_container": [
            'css:div[class*="message"]',
            'css:div[class*="response"]',
            'css:div[class*="answer"]',
        ],
    }
    
    def __init__(self, tab):
        self.tab = tab
        self._cache: Dict[str, CachedElement] = {}
    
    def _compute_element_hash(self, ele) -> str:
        try:
            identity_parts = []
            for attr in ['id', 'data-testid', 'data-message-id']:
                try:
                    val = ele.attr(attr)
                    if val:
                        identity_parts.append(f"{attr}={val}")
                except:
                    pass
            
            try:
                rect = ele.rect
                if rect:
                    identity_parts.append(f"pos={rect.get('x', 0)},{rect.get('y', 0)}")
            except:
                pass
            
            try:
                text_len = len(ele.text or "")
                identity_parts.append(f"len={text_len}")
            except:
                pass
            
            identity_str = "|".join(identity_parts)
            return hashlib.md5(identity_str.encode()).hexdigest()[:8]
        except Exception:
            return ""
    
    def _validate_cached_element(self, cached: CachedElement) -> bool:
        if cached.is_stale():
            return False
        
        ele = cached.element
        try:
            if not (hasattr(ele, 'states') and ele.states.is_displayed):
                return False
            
            current_hash = self._compute_element_hash(ele)
            if current_hash and cached.content_hash:
                if current_hash != cached.content_hash:
                    return False
            
            return True
        except Exception:
            return False
    
    def invalidate_cache(self, selector: str = None):
        if selector:
            self._cache.pop(selector, None)
        else:
            self._cache.clear()
    
    def find(self, selector: str, timeout: int = None,
             use_cache: bool = False) -> Optional[Any]:
        if not selector:
            return None
        
        if timeout is None:
            timeout = BrowserConstants.DEFAULT_ELEMENT_TIMEOUT
        
        if use_cache and selector in self._cache:
            cached = self._cache[selector]
            if self._validate_cached_element(cached):
                return cached.element
            else:
                del self._cache[selector]
        
        try:
            ele = self._find_with_syntax(selector, timeout)
            
            if use_cache and ele:
                self._cache[selector] = CachedElement(
                    element=ele,
                    selector=selector,
                    cached_at=time.time(),
                    content_hash=self._compute_element_hash(ele)
                )
            
            return ele
        except Exception as e:
            if timeout > 1:
                logger.debug(f"元素查找失败 [{selector}]: {e}")
            return None
    
    def find_all(self, selector: str, timeout: int = None) -> List[Any]:
        if not selector:
            return []
        
        if timeout is None:
            timeout = BrowserConstants.DEFAULT_ELEMENT_TIMEOUT
        
        try:
            return self._find_all_with_syntax(selector, timeout)
        except Exception as e:
            logger.debug(f"元素批量查找失败 [{selector}]: {e}")
            return []
    
    def _find_with_syntax(self, selector: str, timeout: int) -> Optional[Any]:
        if selector.startswith(('tag:', '@', 'xpath:', 'css:')) or '@@' in selector:
            ele = self.tab.ele(selector, timeout=timeout)
        else:
            ele = self.tab.ele(f'css:{selector}', timeout=timeout)
        return ele if ele else None
    
    def _find_all_with_syntax(self, selector: str, timeout: int) -> List[Any]:
        try:
            if selector.startswith(('tag:', '@', 'xpath:', 'css:')) or '@@' in selector:
                eles = self.tab.eles(selector, timeout=timeout)
            else:
                eles = self.tab.eles(f'css:{selector}', timeout=timeout)
            return list(eles) if eles else []
        except Exception:
            return []
    
    def find_with_fallback(self, primary_selector: str,
                           target_key: str,
                           timeout: int = None) -> Optional[Any]:
        if primary_selector:
            ele = self.find(primary_selector, timeout)
            if ele:
                return ele
        
        fallback_list = self.FALLBACK_SELECTORS.get(target_key, [])
        if not fallback_list:
            return None
        
        logger.debug(f"主选择器失败，尝试回退: {target_key}")
        
        fallback_timeout = BrowserConstants.FALLBACK_ELEMENT_TIMEOUT
        for fb_selector in fallback_list:
            ele = self.find(fb_selector, fallback_timeout)
            if ele:
                logger.debug(f"回退选择器成功: {fb_selector}")
                return ele
        
        return None


# ================= 流式上下文 =================

class StreamContext:
    """流式监控上下文"""
    
    def __init__(self):
        self.max_seen_text = ""
        self.sent_content_length = 0
        self.baseline_snapshot = None
        self.active_turn_started = False
        self.stable_text_count = 0
        self.last_stable_text = ""
        self.active_turn_baseline_len = 0
        self.sent_text_hash = None
        self.content_version = 0
    
    def calculate_diff(self, current_text: str) -> tuple:
        if not current_text:
            return "", False, None
        
        effective_start = self.active_turn_baseline_len + self.sent_content_length
        
        if len(current_text) > effective_start:
            diff = current_text[effective_start:]
            return diff, False, None
        
        if len(current_text) >= self.active_turn_baseline_len:
            current_active_text = current_text[self.active_turn_baseline_len:]
            
            if len(current_active_text) < self.sent_content_length:
                shrink_amount = self.sent_content_length - len(current_active_text)
                
                if shrink_amount <= BrowserConstants.STREAM_CONTENT_SHRINK_TOLERANCE:
                    return "", False, None
                else:
                    return "", False, f"内容缩短 {shrink_amount} 字符"
        
        if self.max_seen_text and len(self.max_seen_text) > effective_start:
            diff = self.max_seen_text[effective_start:]
            return diff, True, "使用历史快照"
        
        return "", False, None
    
    def update_after_send(self, diff: str, current_text: str):
        self.sent_content_length += len(diff)
        self.last_stable_text = current_text
        self.stable_text_count = 0
        
        if len(current_text) > len(self.max_seen_text):
            self.max_seen_text = current_text


# ================= 生成状态缓存 =================

class GeneratingStatusCache:
    """生成状态缓存"""
    
    def __init__(self, tab):
        self.tab = tab
        self._last_check_time = 0
        self._last_result = False
        self._check_interval = 0.5
        self._found_selector = None
    
    def is_generating(self) -> bool:
        now = time.time()
        
        if now - self._last_check_time < self._check_interval:
            return self._last_result
        
        self._last_check_time = now
        
        if self._found_selector:
            try:
                ele = self.tab.ele(self._found_selector, timeout=0.1)
                if ele and ele.states.is_displayed:
                    self._last_result = True
                    return True
            except:
                pass
            self._found_selector = None
        
        indicator_selectors = [
            'css:button[aria-label*="Stop"]',
            'css:button[aria-label*="stop"]',
            'css:[data-state="streaming"]',
            'css:.stop-generating',
        ]
        
        for selector in indicator_selectors:
            try:
                ele = self.tab.ele(selector, timeout=0.05)
                if ele and ele.states.is_displayed:
                    self._found_selector = selector
                    self._last_result = True
                    return True
            except:
                pass
        
        self._last_result = False
        return False


# ================= 标签页管理器 =================

class TabManager:
    """标签页管理器"""
    
    def __init__(self, page):
        self.page = page
        self._pinned_tab = None
        self._pinned_tab_id = None
    
    def get_active_tab(self, pin: bool = True):
        if self._pinned_tab is not None:
            try:
                if self._is_tab_valid(self._pinned_tab):
                    return self._pinned_tab
            except:
                pass
            
            logger.debug("固定的标签页已失效，重新获取")
            self._pinned_tab = None
            self._pinned_tab_id = None
        
        tab = self.page.latest_tab
        
        if pin:
            self._pinned_tab = tab
            self._pinned_tab_id = self._get_tab_id(tab)
        
        return tab
    
    def release_pinned_tab(self):
        self._pinned_tab = None
        self._pinned_tab_id = None
    
    def _is_tab_valid(self, tab) -> bool:
        try:
            _ = tab.url
            return True
        except:
            return False
    
    def _get_tab_id(self, tab) -> str:
        try:
            return f"{tab.url}#{id(tab)}"
        except:
            return str(id(tab))


# ================= 工作流执行器 =================

class WorkflowExecutor:
    """工作流执行器"""
    
    def __init__(self, tab, stealth_mode: bool = False, 
                 should_stop_checker: Callable[[], bool] = None):
        self.tab = tab
        self.stealth_mode = stealth_mode
        self.finder = ElementFinder(tab)
        self.formatter = SSEFormatter()
        
        # 取消检查器（由外部传入，通常是 RequestContext.should_stop）
        self._should_stop = should_stop_checker or (lambda: False)
        
        self._stream_ctx: Optional[StreamContext] = None
        self._final_complete_text = ""
        self._generating_checker: Optional[GeneratingStatusCache] = None
    
    def _check_cancelled(self) -> bool:
        """检查是否被取消（统一检查点）"""
        return self._should_stop()
    
    def _smart_delay(self, min_sec: float = None, max_sec: float = None):
        """智能延迟（可被取消中断）"""
        if not self.stealth_mode:
            return
        
        min_sec = min_sec or BrowserConstants.STEALTH_DELAY_MIN
        max_sec = max_sec or BrowserConstants.STEALTH_DELAY_MAX
        
        # 分段 sleep，更快响应取消
        total_delay = random.uniform(min_sec, max_sec)
        elapsed = 0
        step = 0.05  # 每 50ms 检查一次
        
        while elapsed < total_delay:
            if self._check_cancelled():
                return
            time.sleep(min(step, total_delay - elapsed))
            elapsed += step
    
    def execute_step(self, action: str, selector: str,
                     target_key: str, value: str = None,
                     optional: bool = False,
                     context: Dict = None) -> Generator[str, None, None]:
        """执行单个步骤"""
        
        # 每步开始检查取消
        if self._check_cancelled():
            logger.debug(f"步骤 {action} 跳过（已取消）")
            return
        
        logger.debug(f"执行: {action} -> {target_key}")
        
        try:
            if action == "WAIT":
                # 可中断的等待
                wait_time = float(value or 0.5)
                elapsed = 0
                while elapsed < wait_time:
                    if self._check_cancelled():
                        return
                    time.sleep(min(0.1, wait_time - elapsed))
                    elapsed += 0.1
            
            elif action == "KEY_PRESS":
                self._execute_keypress(target_key or value)
            
            elif action == "CLICK":
                self._execute_click(selector, target_key, optional)
            
            elif action == "FILL_INPUT":
                prompt = context.get("prompt", "") if context else ""
                self._execute_fill(selector, prompt, target_key, optional)
            
            elif action in ("STREAM_WAIT", "STREAM_OUTPUT"):
                yield from self._stream_monitor(selector)
            
            else:
                logger.debug(f"未知动作: {action}")
        
        except ElementNotFoundError as e:
            if not optional:
                yield self.formatter.pack_error(f"元素未找到: {str(e)}")
                raise
        
        except Exception as e:
            logger.error(f"步骤执行失败 [{action}]: {e}")
            if not optional:
                yield self.formatter.pack_error(f"执行失败: {str(e)}")
                raise
    
    def _execute_keypress(self, key: str):
        if self._check_cancelled():
            return
        self.tab.actions.key_down(key).key_up(key)
        self._smart_delay(0.1, 0.2)
    
    def _execute_click(self, selector: str, target_key: str, optional: bool):
        if self._check_cancelled():
            return
        
        ele = self.finder.find_with_fallback(selector, target_key)
        
        if ele:
            try:
                if self.stealth_mode:
                    try:
                        self.tab.actions.move_to(ele)
                        self._smart_delay(0.1, 0.25)
                    except Exception:
                        pass
                
                if self._check_cancelled():
                    return
                
                ele.click()
                self._smart_delay(
                    BrowserConstants.ACTION_DELAY_MIN,
                    BrowserConstants.ACTION_DELAY_MAX
                )
            
            except Exception as click_err:
                logger.debug(f"点击异常: {click_err}")
                if target_key == "send_btn":
                    self._execute_keypress("Enter")
        
        elif target_key == "send_btn":
            self._execute_keypress("Enter")
        
        elif not optional:
            raise ElementNotFoundError(f"点击目标未找到: {selector}")
    
    def _execute_fill(self, selector: str, text: str,
                      target_key: str, optional: bool):
        if self._check_cancelled():
            return
        
        ele = self.finder.find_with_fallback(selector, target_key)
        
        if not ele:
            if not optional:
                raise ElementNotFoundError("找不到输入框")
            return
        
        if self.stealth_mode:
            try:
                self.tab.actions.move_to(ele)
                self._smart_delay(0.1, 0.2)
                ele.click()
                self._smart_delay(0.15, 0.25)
            except Exception:
                pass
        
        if self._check_cancelled():
            return
        
        try:
            ele.clear()
        except Exception:
            pass
        
        ele.input(text)
        logger.debug(f"文本已输入 ({len(text)} 字符)")
        self._smart_delay(0.1, 0.2)
        
        try:
            ele.input(' ')
            time.sleep(0.05)
            self.tab.actions.key_down('Backspace').key_up('Backspace')
        except Exception:
            pass

    # ==================== 流式监听 ====================

    def _read_visible_text(self, ele) -> str:
        """读取元素的可见文本（保留格式）"""
        if not ele:
            return ""

        try:
            text = ele.run_js("""
                function getTextWithWhitespace(element) {
                    let text = '';
                
                    function processNode(node) {
                        if (node.nodeType === Node.TEXT_NODE) {
                            // 文本节点：直接获取原始值
                            text += node.nodeValue;
                        } else if (node.nodeType === Node.ELEMENT_NODE) {
                            const tagName = node.tagName.toLowerCase();
                            const style = window.getComputedStyle(node);
                        
                            // 跳过隐藏元素
                            if (style.display === 'none' || style.visibility === 'hidden') {
                                return;
                            }
                        
                            // 块级元素前添加换行
                            const blockTags = ['div', 'p', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
                                              'li', 'tr', 'blockquote', 'pre', 'section', 'article'];
                        
                            if (tagName === 'br') {
                                text += '\\n';
                                return;
                            }
                        
                            const isBlock = blockTags.includes(tagName) || 
                                            style.display === 'block';
                        
                            if (isBlock && text && !text.endsWith('\\n')) {
                                text += '\\n';
                            }
                        
                            // 递归处理子节点
                            for (const child of node.childNodes) {
                                processNode(child);
                            }
                        
                            // 块级元素后添加换行
                            if (isBlock && text && !text.endsWith('\\n')) {
                                text += '\\n';
                            }
                        }
                    }
                
                    processNode(element);
                    return text;
                }
            
                return getTextWithWhitespace(this);
            """)
        
            if text:
                # 清理多余的连续换行，但保留单个换行
                import re
                text = re.sub(r'\n{3,}', '\n\n', str(text))
                return text.strip()
            
        except Exception as e:
            pass

        # 降级方案
        try:
            if hasattr(ele, 'text') and ele.text:
                return str(ele.text)
        except Exception:
            pass

        return ""

    def _get_message_anchor(self, element) -> str:
        if not element:
            return ""
        
        try:
            stable_attrs = ['data-message-id', 'data-turn-id', 'data-testid', 'id']
            
            for attr in stable_attrs:
                try:
                    val = element.attr(attr)
                    if val:
                        return f"{attr}={val}"
                except Exception:
                    pass
            
            try:
                tag = element.tag if hasattr(element, 'tag') else 'unknown'
                cls = element.attr('class') or ''
                classes = cls.split()[:2]
                if classes:
                    return f"tag:{tag}|class:{'.'.join(classes)}"
                return f"tag:{tag}"
            except Exception:
                pass
            
            return ""
        except Exception:
            return ""

    def _get_latest_message_snapshot(self, selector: str) -> dict:
        result = {
            'groups_count': 0,
            'anchor': None,
            'text': '',
            'text_len': 0,
            'is_generating': False,
        }
        
        try:
            eles = self.finder.find_all(selector, timeout=1)
            
            if not eles:
                return result
            
            result['groups_count'] = len(eles)
            
            last_ele = eles[-1]
            result['text'] = self._read_visible_text(last_ele)
            result['text_len'] = len(result['text'])
            result['anchor'] = self._get_message_anchor(last_ele)
            
            if self._generating_checker is None:
                self._generating_checker = GeneratingStatusCache(self.tab)
            result['is_generating'] = self._generating_checker.is_generating()
        
        except Exception as e:
            logger.debug(f"Baseline 异常: {e}")
        
        return result

    def _detect_new_turn(self, baseline: dict, current: dict) -> tuple:
        if current['groups_count'] > baseline['groups_count']:
            return True, "元素数量增加", False

        if current['groups_count'] == 0 and baseline['groups_count'] > 0:
            return False, "元素暂时消失", True

        if baseline['text_len'] > 0 and current['text_len'] > 0:
            if current['groups_count'] == baseline['groups_count']:
                shrink_ratio = current['text_len'] / baseline['text_len']
                if shrink_ratio < 0.3:
                    return False, f"内容大幅减少({shrink_ratio:.0%})", True

        if baseline['anchor'] and current['anchor']:
            if baseline['anchor'] != current['anchor']:
                if current['groups_count'] == baseline['groups_count']:
                    return False, "消息锚点变化", True

        if current['is_generating'] and not baseline['is_generating']:
            return True, "生成指示器激活", False

        if baseline['text_len'] == 0 and current['text_len'] > 0:
            if current['is_generating']:
                return True, "内容出现+生成中", False
            return False, "", False

        if current['text_len'] > baseline['text_len']:
            growth = current['text_len'] - baseline['text_len']

            if current['is_generating']:
                return True, f"内容增长({growth}字符)+生成中", False

            if growth > 50:
                return True, f"显著内容增长({growth}字符)", False

        return False, "", False

    def _get_active_turn_text(self, selector: str) -> str:
        try:
            eles = self.finder.find_all(selector, timeout=1)
            
            if not eles:
                return ""
            
            for i in range(len(eles) - 1, -1, -1):
                ele = eles[i]
                text = self._read_visible_text(ele)
                
                if text and text.strip():
                    return text.strip()
            
            return ""
        except Exception:
            return ""

    def _stream_monitor(self, selector: str) -> Generator[str, None, None]:
        """流式监听（增强取消检查）"""
        logger.debug("流式监听启动")

        ctx = StreamContext()
        self._stream_ctx = ctx
        self._generating_checker = GeneratingStatusCache(self.tab)

        # 等待初始元素
        baseline = None
        initial_wait_start = time.time()
        initial_wait_max = 10

        while time.time() - initial_wait_start < initial_wait_max:
            if self._check_cancelled():
                logger.info("等待初始元素时被取消")
                return
        
            baseline = self._get_latest_message_snapshot(selector)
            if baseline['groups_count'] > 0:
                break
            time.sleep(0.2)

        if baseline['groups_count'] == 0:
            logger.debug("初始元素未出现")

        ctx.baseline_snapshot = baseline

        start_time = time.time()
        silence_start = time.time()
        last_output_time = None

        current_interval = BrowserConstants.STREAM_CHECK_INTERVAL_DEFAULT
        min_interval = BrowserConstants.STREAM_CHECK_INTERVAL_MIN
        max_interval = BrowserConstants.STREAM_CHECK_INTERVAL_MAX

        abnormal_count = 0
        max_abnormal_count = 5
        element_missing_count = 0
        max_element_missing = 10
    
        # ★ 修复：初始化 current_text
        current_text = ""

        while True:
            if self._check_cancelled():
                logger.info("流式监听被取消")
                break
        
            elapsed = time.time() - start_time
        
            # === 未进入 Active Turn ===
            if not ctx.active_turn_started:
                try:
                    current = self._get_latest_message_snapshot(selector)
                except Exception:
                    current = {'groups_count': 0, 'anchor': None, 'text': '',
                               'text_len': 0, 'is_generating': False}
        
                is_new_turn, reason, is_abnormal = self._detect_new_turn(baseline, current)
        
                if is_abnormal:
                    abnormal_count += 1
            
                    if current['groups_count'] == 0:
                        element_missing_count += 1
                        if element_missing_count >= max_element_missing:
                            yield self.formatter.pack_error("页面元素丢失", code="element_lost")
                            return
                        time.sleep(0.3)
                        continue
                    else:
                        element_missing_count = 0
            
                    if abnormal_count >= max_abnormal_count:
                        if "减少" in reason or "锚点" in reason:
                            baseline = current
                            ctx.baseline_snapshot = baseline
                            abnormal_count = 0
                    
                            if current['text_len'] > 0 and current['is_generating']:
                                ctx.active_turn_started = True
                                ctx.active_turn_baseline_len = 0
                                current_text = current['text']  # ★ 修复
                            continue
                        else:
                            yield self.formatter.pack_error(f"状态异常: {reason}", code="abnormal_state")
                            return
                else:
                    abnormal_count = 0
                    element_missing_count = 0
        
                if is_new_turn:
                    ctx.active_turn_started = True
                    logger.debug(f"检测到新回复 (原因: {reason})")
            
                    if current['groups_count'] > baseline['groups_count']:
                        ctx.active_turn_baseline_len = 0
                    else:
                        ctx.active_turn_baseline_len = baseline['text_len']
                
                    # ★ 修复：进入 Active Turn 时设置 current_text
                    current_text = current['text']
                else:
                    if elapsed > BrowserConstants.STREAM_INITIAL_WAIT:
                        logger.debug("等待超时，未检测到新回复")
                        break
            
                    time.sleep(current_interval)
                    continue
    
            # === 已进入 Active Turn ===
            else:
                try:
                    current_text = self._get_active_turn_text(selector)
                    if not current_text and ctx.max_seen_text:
                        current_text = ctx.max_seen_text
                except Exception:
                    current_text = ctx.max_seen_text if ctx.max_seen_text else ""
        
                if not current_text and ctx.sent_content_length > 0:
                    element_missing_count += 1
                    if element_missing_count >= max_element_missing:
                        break
                    time.sleep(0.2)
                    continue
                else:
                    element_missing_count = 0
    
            # === 计算增量 ===
            if current_text and len(current_text) > len(ctx.max_seen_text):
                ctx.max_seen_text = current_text
    
            diff, needs_resync, resync_reason = ctx.calculate_diff(current_text)
    
            if diff:
                if self._check_cancelled():
                    logger.info("发送增量前被取消")
                    break
            
                ctx.update_after_send(diff, current_text)
                silence_start = time.time()
                if last_output_time is None:
                    last_output_time = time.time()
                current_interval = min_interval
        
                yield self.formatter.pack_chunk(diff)
            else:
                if current_text == ctx.last_stable_text:
                    ctx.stable_text_count += 1
                else:
                    ctx.stable_text_count = 0
                    ctx.last_stable_text = current_text
        
                current_interval = min(current_interval * 1.5, max_interval)
    
            silence_duration = time.time() - silence_start
            still_generating = self._generating_checker.is_generating()
    
            # === 结束判定 ===
            if last_output_time is not None:
                if not still_generating:
                    if silence_duration > BrowserConstants.STREAM_SILENCE_THRESHOLD:
                        logger.debug("生成结束 (指示器消失 + 静默)")
                        break
                else:
                    if (ctx.stable_text_count >= BrowserConstants.STREAM_STABLE_COUNT_THRESHOLD and
                        silence_duration > BrowserConstants.STREAM_SILENCE_THRESHOLD_FALLBACK):
                        logger.debug("生成结束 (稳定 + 静默)")
                        break
    
            if elapsed > BrowserConstants.STREAM_MAX_TIMEOUT:
                logger.debug("超时退出")
                break
    
            # 分段 sleep
            sleep_elapsed = 0
            while sleep_elapsed < current_interval:
                if self._check_cancelled():
                    break
                time.sleep(min(0.1, current_interval - sleep_elapsed))
                sleep_elapsed += 0.1

        # === 最终读取 ===
        if not self._check_cancelled():
            try:
                final_text = self._get_active_turn_text(selector)
                if final_text:
                    final_effective_start = ctx.active_turn_baseline_len + ctx.sent_content_length
                    if len(final_text) > final_effective_start:
                        remaining = final_text[final_effective_start:]
                        if remaining.strip():
                            yield self.formatter.pack_chunk(remaining)
                            ctx.sent_content_length += len(remaining)
        
                    self._final_complete_text = final_text[ctx.active_turn_baseline_len:]
                else:
                    self._final_complete_text = ctx.max_seen_text[ctx.active_turn_baseline_len:] if ctx.max_seen_text else ""
            except Exception:
                self._final_complete_text = ctx.max_seen_text[ctx.active_turn_baseline_len:] if ctx.max_seen_text else ""

# ================= 浏览器核心 =================

class BrowserCore:
    """
    浏览器核心类 - 单例模式
    
    重构说明：
    - 移除内部锁管理，由 RequestManager 统一控制
    - 接收外部的取消信号检查器
    """
    
    _instance: Optional['BrowserCore'] = None
    _lock = threading.Lock()
    
    def __new__(cls, port: int = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance
    
    def __init__(self, port: int = None):
        if self._initialized:
            return
        
        self.port = port or BrowserConstants.DEFAULT_PORT
        self.page: Optional[ChromiumPage] = None
        self.stealth_mode = False
        
        self._connected = False
        
        # 外部传入的停止检查器
        self._should_stop_checker: Callable[[], bool] = lambda: False
        
        self.formatter = SSEFormatter()
        self.config_engine = None
        
        self._tab_manager: Optional[TabManager] = None
        
        self._initialized = True
    
    def set_stop_checker(self, checker: Callable[[], bool]):
        """
        设置停止检查器
        
        由 main.py 在执行前设置，通常是 RequestContext.should_stop
        """
        self._should_stop_checker = checker or (lambda: False)
    
    @property
    def tab_manager(self) -> TabManager:
        if self._tab_manager is None:
            if not self.ensure_connection():
                raise BrowserConnectionError("无法连接到浏览器")
            self._tab_manager = TabManager(self.page)
        return self._tab_manager
    
    def _get_config_engine(self):
        if self.config_engine is None:
            from config_engine import config_engine
            self.config_engine = config_engine
        return self.config_engine
    
    def _connect(self) -> bool:
        try:
            logger.debug(f"连接浏览器 127.0.0.1:{self.port}")
            self.page = ChromiumPage(addr_or_opts=f"127.0.0.1:{self.port}")
            self._connected = True
            logger.info("浏览器连接成功")
            return True
        except Exception as e:
            logger.error(f"浏览器连接失败: {e}")
            self._connected = False
            return False
    
    def health_check(self) -> Dict[str, Any]:
        result = {
            "status": "unhealthy",
            "connected": False,
            "port": self.port,
            "tab_url": None,
            "tab_title": None,
            "error": None
        }
        
        try:
            if not self.page:
                if not self._connect():
                    result["error"] = "无法连接到浏览器"
                    return result
            
            tab = self.page.latest_tab
            if not tab:
                result["error"] = "无可用标签页"
                return result
            
            result["status"] = "healthy"
            result["connected"] = True
            result["tab_url"] = tab.url
            result["tab_title"] = tab.title
        
        except Exception as e:
            result["error"] = str(e)
            self._connected = False
        
        return result
    
    def ensure_connection(self) -> bool:
        if self._connected:
            try:
                _ = self.page.latest_tab
                return True
            except Exception:
                self._connected = False
        
        return self._connect()
    
    def get_active_tab(self):
        if not self.ensure_connection():
            raise BrowserConnectionError("无法连接到浏览器")
        return self.tab_manager.get_active_tab(pin=True)
    
    def execute_workflow(self, messages: List[Dict],
                         stream: bool = True) -> Generator[str, None, None]:
        """
        工作流执行入口
        
        注意：
        - 锁的获取/释放由调用方（main.py）通过 RequestManager 管理
        - 取消检查器需要提前通过 set_stop_checker 设置
        """
        # 验证输入
        is_valid, error_msg, sanitized_messages = MessageValidator.validate(messages)
        
        if not is_valid:
            yield self.formatter.pack_error(
                f"无效请求: {error_msg}",
                error_type="invalid_request_error",
                code="invalid_messages"
            )
            return
        
        # 执行工作流
        if stream:
            yield from self._execute_workflow_stream(sanitized_messages)
        else:
            yield from self._execute_workflow_non_stream(sanitized_messages)
    
    def _execute_workflow_stream(self, messages: List[Dict]) -> Generator[str, None, None]:
        """流式工作流执行"""
        
        # 检查取消
        if self._should_stop_checker():
            yield self.formatter.pack_error("请求已取消", code="cancelled")
            yield self.formatter.pack_finish()
            return
        
        if not self.ensure_connection():
            yield self.formatter.pack_error(
                "浏览器连接失败",
                error_type="connection_error",
                code="browser_disconnected"
            )
            yield self.formatter.pack_finish()
            return
        
        try:
            tab = self.get_active_tab()
        except BrowserConnectionError as e:
            yield self.formatter.pack_error(str(e))
            yield self.formatter.pack_finish()
            return
        
        try:
            domain = tab.url.split("//")[-1].split("/")[0]
        except Exception:
            yield self.formatter.pack_error("无法解析页面URL")
            yield self.formatter.pack_finish()
            return
        
        logger.debug(f"域名: {domain}")
        
        page_status = self._check_page_status(tab)
        if not page_status["ready"]:
            yield self.formatter.pack_error(
                f"页面未就绪: {page_status['reason']}",
                code="page_not_ready"
            )
            yield self.formatter.pack_finish()
            return
        
        config_engine = self._get_config_engine()
        site_config = config_engine.get_site_config(domain, tab.html)
        if not site_config:
            yield self.formatter.pack_error(
                "配置加载失败",
                code="config_error"
            )
            yield self.formatter.pack_finish()
            return
        
        selectors = site_config.get("selectors", {})
        workflow = site_config.get("workflow", [])
        self.stealth_mode = site_config.get("stealth", False)
        
        context = {
            "prompt": "\n\n".join([
                f"{m.get('role', 'user')}: {m.get('content', '')}"
                for m in messages
            ])
        }
        
        # 创建执行器，传入取消检查器
        executor = WorkflowExecutor(
            tab, 
            self.stealth_mode,
            should_stop_checker=self._should_stop_checker
        )
        
        try:
            for step in workflow:
                # 每步前检查取消
                if self._should_stop_checker():
                    logger.info("工作流被用户中断")
                    break
                
                action = step.get('action', '')
                target_key = step.get('target', '')
                optional = step.get('optional', False)
                param_value = step.get('value')
                
                selector = selectors.get(target_key, '')
                
                if not selector and action not in ("WAIT", "KEY_PRESS"):
                    if optional:
                        continue
                    else:
                        yield self.formatter.pack_error(
                            f"缺少配置: {target_key}",
                            code="missing_selector"
                        )
                        break
                
                try:
                    yield from executor.execute_step(
                        action=action,
                        selector=selector,
                        target_key=target_key,
                        value=param_value,
                        optional=optional,
                        context=context
                    )
                except (ElementNotFoundError, WorkflowError):
                    break
                except Exception as e:
                    if not optional:
                        yield self.formatter.pack_error(f"执行中断: {str(e)}")
                        break
        
        finally:
            yield self.formatter.pack_finish()
            self.tab_manager.release_pinned_tab()
    
    def _execute_workflow_non_stream(self, messages: List[Dict]) -> Generator[str, None, None]:
        """非流式工作流执行"""
        collected_content = []
        error_data = None
        
        for chunk in self._execute_workflow_stream(messages):
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
            yield json.dumps(error_data, ensure_ascii=False)
        else:
            full_content = "".join(collected_content)
            response = self.formatter.pack_non_stream(full_content)
            yield json.dumps(response, ensure_ascii=False)

    def _check_page_status(self, tab) -> Dict[str, Any]:
        """检查页面状态"""
        result = {"ready": True, "reason": None}
        
        try:
            url = tab.url or ""
            
            if not url or url in ("about:blank", "chrome://newtab/"):
                result["ready"] = False
                result["reason"] = "请先打开目标AI网站"
                return result
            
            error_indicators = ["chrome-error://", "about:neterror"]
            for indicator in error_indicators:
                if indicator in url:
                    result["ready"] = False
                    result["reason"] = "页面加载错误"
                    return result
            
            title = (tab.title or "").lower()
            warning_keywords = ["404", "not found", "error", "无法访问", "refused"]
            for keyword in warning_keywords:
                if keyword in title:
                    logger.debug(f"页面可能存在问题: {tab.title}")
                    break
        
        except Exception as e:
            logger.debug(f"页面状态检查异常: {e}")
        
        return result
    
    def close(self):
        """关闭浏览器连接"""
        logger.info("关闭浏览器连接")
        
        self._connected = False
        self.page = None
        self._tab_manager = None
        
        with self._lock:
            BrowserCore._instance = None
            self._initialized = False


# ================= 工厂函数 =================

_browser_instance: Optional[BrowserCore] = None
_browser_lock = threading.Lock()


def get_browser(port: int = None, auto_connect: bool = True) -> BrowserCore:
    """获取浏览器实例（线程安全延迟初始化）"""
    global _browser_instance
    
    if _browser_instance is not None:
        return _browser_instance
    
    with _browser_lock:
        if _browser_instance is None:
            instance = BrowserCore(port)
            
            if auto_connect:
                if not instance.ensure_connection():
                    raise BrowserConnectionError(
                        f"无法连接到浏览器 (端口: {instance.port})"
                    )
            
            _browser_instance = instance
    
    return _browser_instance


class _LazyBrowser:
    """浏览器延迟初始化代理"""
    
    def __getattr__(self, name):
        return getattr(get_browser(auto_connect=False), name)
    
    def __call__(self, *args, **kwargs):
        return get_browser(*args, **kwargs)


# 向后兼容的模块级实例
browser = _LazyBrowser()
