"""
browser_core.py - 浏览器自动化核心模块

职责：
- 浏览器连接管理
- 工作流执行
- 元素查找与操作

重构说明（v4）：
- 彻底解决极速AI回复的180s延迟问题
- 引入两阶段 baseline 机制（instant + user）
- 明确的状态机：等待用户消息 -> 等待AI开始 -> 正常输出
- 完全不依赖 pre-send 记录（避免路径依赖）
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
        'STREAM_SILENCE_THRESHOLD': 3.0,
        'STREAM_MAX_TIMEOUT': 600,
        'STREAM_INITIAL_WAIT': 180,
        'STREAM_RERENDER_WAIT': 0.5,
        'STREAM_CONTENT_SHRINK_TOLERANCE': 3,
        'STREAM_MIN_VALID_LENGTH': 10,
        'STREAM_STABLE_COUNT_THRESHOLD': 5,
        'STREAM_SILENCE_THRESHOLD_FALLBACK': 6.0,
        'MAX_MESSAGE_LENGTH': 100000,
        'MAX_MESSAGES_COUNT': 100,
        'STREAM_INITIAL_ELEMENT_WAIT': 10,
        'STREAM_MAX_ABNORMAL_COUNT': 5,
        'STREAM_MAX_ELEMENT_MISSING': 10,
        'STREAM_CONTENT_SHRINK_THRESHOLD': 0.3,
        'STREAM_USER_MSG_WAIT': 1.5,  # v4 新增：等待用户消息上屏的最大时间
        'STREAM_PRE_BASELINE_DELAY': 0.3,  # v4 新增：instant baseline 后的延迟
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
    
    # 流式监控
    STREAM_CHECK_INTERVAL_MIN = 0.1
    STREAM_CHECK_INTERVAL_MAX = 1.0
    STREAM_CHECK_INTERVAL_DEFAULT = 0.3
    
    STREAM_SILENCE_THRESHOLD = 3.0
    STREAM_MAX_TIMEOUT = 600
    STREAM_INITIAL_WAIT = 180  # 保持高值以兼容长思考AI
    
    # 流式监控增强配置
    STREAM_RERENDER_WAIT = 0.5
    STREAM_CONTENT_SHRINK_TOLERANCE = 3
    STREAM_MIN_VALID_LENGTH = 10
    
    STREAM_STABLE_COUNT_THRESHOLD = 5
    STREAM_SILENCE_THRESHOLD_FALLBACK = 6.0
    
    # 输入验证
    MAX_MESSAGE_LENGTH = 100000
    MAX_MESSAGES_COUNT = 100

    # 异常检测配置
    STREAM_INITIAL_ELEMENT_WAIT = 10
    STREAM_MAX_ABNORMAL_COUNT = 5
    STREAM_MAX_ELEMENT_MISSING = 10
    STREAM_CONTENT_SHRINK_THRESHOLD = 0.3
    
    # v4 新增：两阶段 baseline 配置
    STREAM_USER_MSG_WAIT = 1.5  # 等待用户消息上屏的最大时间
    STREAM_PRE_BASELINE_DELAY = 0.3  # instant baseline 后的延迟

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
    """流式监控上下文（v4 优化版）"""
    
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
        
        # v4 新增：两阶段 baseline
        self.instant_baseline = None  # 进入监控瞬间的快照
        self.user_baseline = None     # 等待用户消息上屏后的快照
        
        # v4 新增：状态标记
        self.content_ever_changed = False  # 是否曾经产出过内容
        self.user_msg_confirmed = False    # 用户消息是否已确认上屏
        self.output_target_anchor = None      # 当前锁定的输出目标节点锚点
        self.output_target_count = 0          # 锁定时的节点总数
        self.pending_new_anchor = None        # 候选的新节点锚点（用于确认）
        self.pending_new_anchor_seen = 0      # 候选新节点连续出现次数
    def reset_for_new_target(self):
        """切换到新目标节点时重置状态"""
        self.max_seen_text = ""
        self.sent_content_length = 0
        self.stable_text_count = 0
        self.last_stable_text = ""
        self.active_turn_baseline_len = 0
        self.content_ever_changed = False
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
    """工作流执行器（v4 优化版）"""
    
    def __init__(self, tab, stealth_mode: bool = False, 
                 should_stop_checker: Callable[[], bool] = None):
        self.tab = tab
        self.stealth_mode = stealth_mode
        self.finder = ElementFinder(tab)
        self.formatter = SSEFormatter()
        
        # 取消检查器（由外部传入）
        self._should_stop = should_stop_checker or (lambda: False)
        
        self._stream_ctx: Optional[StreamContext] = None
        self._final_complete_text = ""
        self._generating_checker: Optional[GeneratingStatusCache] = None
    
    def _check_cancelled(self) -> bool:
        """检查是否被取消"""
        return self._should_stop()
    
    def _smart_delay(self, min_sec: float = None, max_sec: float = None):
        """智能延迟（可被取消中断）"""
        if not self.stealth_mode:
            return
        
        min_sec = min_sec or BrowserConstants.STEALTH_DELAY_MIN
        max_sec = max_sec or BrowserConstants.STEALTH_DELAY_MAX
        
        total_delay = random.uniform(min_sec, max_sec)
        elapsed = 0
        step = 0.05
        
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
        
        if self._check_cancelled():
            logger.debug(f"步骤 {action} 跳过（已取消）")
            return
        
        logger.debug(f"执行: {action} -> {target_key}")
        
        try:
            if action == "WAIT":
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
                            text += node.nodeValue;
                        } else if (node.nodeType === Node.ELEMENT_NODE) {
                            const tagName = node.tagName.toLowerCase();
                            const style = window.getComputedStyle(node);
                        
                            if (style.display === 'none' || style.visibility === 'hidden') {
                                return;
                            }
                        
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
                        
                            for (const child of node.childNodes) {
                                processNode(child);
                            }
                        
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
                import re
                text = re.sub(r'\n{3,}', '\n\n', str(text))
                return text.strip()
            
        except Exception:
            pass

        try:
            if hasattr(ele, 'text') and ele.text:
                return str(ele.text)
        except Exception:
            pass

        return ""

    def _get_message_anchor(self, element) -> str:
        """获取消息元素的唯一标识"""
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
        """获取最新消息快照"""
        result = {
            'groups_count': 0,
            'anchor': None,
            'text': '',
            'text_len': 0,
            'is_generating': False,
        }
        
        try:
            eles = self.finder.find_all(selector, timeout=0.5)
            
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
            logger.debug(f"Snapshot 异常: {e}")
        
        return result

    def _get_active_turn_text(self, selector: str) -> str:
        """获取当前活跃轮次的文本"""
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
        """
        流式监听（v4 完全重写版）
        
        核心策略：
        1. 进入监控瞬间抓 instant_baseline
        2. 等待短暂时间（让用户消息上屏）
        3. 抓 user_baseline，对比两者：
           - 如果 count 增加 1：用户消息已上屏，继续等 AI
           - 如果 count 增加 2+：AI 秒回，直接进入输出
        4. 进入正常监控阶段
        """
        logger.info("========== 流式监听启动 (v4) ==========")

        ctx = StreamContext()
        self._stream_ctx = ctx
        self._generating_checker = GeneratingStatusCache(self.tab)

        # ===== 阶段 0：抓取 instant baseline =====
        logger.debug("[阶段0] 抓取 instant baseline（监控启动瞬间）")
        ctx.instant_baseline = self._get_latest_message_snapshot(selector)
        logger.info(f"[Instant] count={ctx.instant_baseline['groups_count']}, "
                   f"text_len={ctx.instant_baseline['text_len']}, "
                   f"generating={ctx.instant_baseline['is_generating']}")
        
        # ===== 阶段 1：等待用户消息上屏 =====
        logger.debug(f"[阶段1] 等待用户消息上屏（最多 {BrowserConstants.STREAM_USER_MSG_WAIT}s）")
        user_msg_wait_start = time.time()
        user_msg_wait_max = BrowserConstants.STREAM_USER_MSG_WAIT
        
        ctx.user_baseline = None
        
        while time.time() - user_msg_wait_start < user_msg_wait_max:
            if self._check_cancelled():
                logger.info("等待用户消息时被取消")
                return
            
            current_snapshot = self._get_latest_message_snapshot(selector)
            current_count = current_snapshot['groups_count']
            instant_count = ctx.instant_baseline['groups_count']
            
            # 情况 A：用户消息已上屏（count 增加 1）
            if current_count == instant_count + 1:
                logger.info(f"[User Msg] 检测到用户消息上屏 (count: {instant_count} -> {current_count})")
                ctx.user_baseline = current_snapshot
                ctx.user_msg_confirmed = True
                break
            
            # 情况 B：AI 秒回（count 增加 2+）
            elif current_count >= instant_count + 2:
                logger.info(f"[Fast AI] 检测到 AI 秒回！(count: {instant_count} -> {current_count})")
                ctx.user_baseline = current_snapshot
                ctx.user_msg_confirmed = True
                # 直接激活 active_turn
                ctx.active_turn_started = True
                ctx.active_turn_baseline_len = 0
                break
            
            # 情况 C：count 没变，但文本增长了（某些网站不增加节点）
            elif current_count == instant_count:
                if current_snapshot['text_len'] > ctx.instant_baseline['text_len'] + 10:
                    logger.info(f"[Same Node] 同节点检测到文本增长，可能为AI回复")
                    ctx.user_baseline = current_snapshot
                    ctx.user_msg_confirmed = True
                    ctx.active_turn_started = True
                    ctx.active_turn_baseline_len = ctx.instant_baseline['text_len']
                    break
            
            time.sleep(0.2)
        
        # 如果超时还没检测到用户消息，使用 instant_baseline 作为 user_baseline
        if ctx.user_baseline is None:
            logger.warning("[Timeout] 未检测到用户消息上屏，使用 instant baseline")
            ctx.user_baseline = ctx.instant_baseline
        
        # ===== 阶段 2：等待 AI 开始回复（如果还没开始）=====
        if not ctx.active_turn_started:
            logger.debug("[阶段2] 等待 AI 开始回复")
            
            baseline = ctx.user_baseline
            start_time = time.time()
            
            while True:
                if self._check_cancelled():
                    logger.info("等待AI开始时被取消")
                    return
                
                elapsed = time.time() - start_time
                
                current = self._get_latest_message_snapshot(selector)
                
                # 检测 AI 是否开始
                is_started, reason = self._detect_ai_start(baseline, current)
                
                if is_started:
                    logger.info(f"[AI Start] 检测到 AI 开始回复：{reason}")
                    ctx.active_turn_started = True
                    
                    # 根据节点数决定 baseline 长度
                    if current['groups_count'] > baseline['groups_count']:
                        ctx.active_turn_baseline_len = 0
                    else:
                        ctx.active_turn_baseline_len = baseline.get('text_len', 0)
                    
                    break
                
                # 超时检查
                if elapsed > BrowserConstants.STREAM_INITIAL_WAIT:
                    logger.warning(f"[Timeout] 等待 AI 开始超时（{elapsed:.1f}s）")
                    break
                
                time.sleep(0.3)
        
        # ===== 阶段 3：正常增量输出 =====
        if ctx.active_turn_started:
            logger.debug("[阶段3] 进入增量输出阶段")
            yield from self._stream_output_phase(selector, ctx)
        else:
            logger.warning("[Exit] 未检测到 AI 回复，退出监控")

    def _detect_ai_start(self, baseline: dict, current: dict) -> tuple:
        """
        检测 AI 是否开始回复（v4 简化版）
        
        返回: (is_started: bool, reason: str)
        """
        
        # 优先级 1：节点数增加
        if current['groups_count'] > baseline['groups_count']:
            increase = current['groups_count'] - baseline['groups_count']
            return True, f"节点数增加 {increase}"
        
        # 优先级 2：生成指示器出现
        if current['is_generating']:
            return True, "生成指示器激活"
        
        # 优先级 3：文本显著增长（同节点）
        if current['text_len'] > baseline['text_len'] + 10:
            growth = current['text_len'] - baseline['text_len']
            return True, f"文本增长 {growth} 字符"
        
        return False, ""

    def _stream_output_phase(self, selector: str, ctx: StreamContext) -> Generator[str, None, None]:
        """
        增量输出阶段（v4.2 优化版）
    
        核心改进：
        - 检测新节点出现时自动切换输出目标
        - 支持"思考 -> 最终答复"模式（节点数增加）
        - 支持"同节点内折叠思考"模式（文本突然变短）
        """
    
        silence_start = time.time()
        last_output_time = None
    
        current_interval = BrowserConstants.STREAM_CHECK_INTERVAL_DEFAULT
        min_interval = BrowserConstants.STREAM_CHECK_INTERVAL_MIN
        max_interval = BrowserConstants.STREAM_CHECK_INTERVAL_MAX
    
        element_missing_count = 0
        max_element_missing = 10
    
        current_text = ""
        last_text_len = 0
    
        # v4.1：记录初始节点数和锚点
        initial_snap = self._get_last_message_snapshot(selector)
        ctx.output_target_count = initial_snap['groups_count']
        ctx.output_target_anchor = initial_snap['anchor']
        logger.debug(f"[Output] 初始目标: count={ctx.output_target_count}, anchor={ctx.output_target_anchor}")
    
        # v4.2 新增：记录峰值文本长度，用于检测"折叠"
        peak_text_len = 0
        content_shrink_count = 0
    
        while True:
            if self._check_cancelled():
                logger.info("输出阶段被取消")
                break
        
            # v4.1：读取最后节点快照
            snap = self._get_last_message_snapshot(selector)
            current_count = snap['groups_count']
            current_anchor = snap['anchor']
            current_text = snap['text'] or ""
            still_generating = snap['is_generating']
            current_text_len = len(current_text)
        
            # ===== v4.2 新增：检测"同节点内折叠"（文本突然大幅变短）=====
            if current_text_len > peak_text_len:
                peak_text_len = current_text_len
                content_shrink_count = 0
            elif peak_text_len > 100 and current_text_len < peak_text_len * 0.5:
                # 文本长度下降超过 50%，可能是思考被折叠了
                content_shrink_count += 1
                if content_shrink_count >= 2:  # 连续 2 次确认
                    logger.info(f"[Collapse] 检测到内容折叠：{peak_text_len} -> {current_text_len}，重置输出状态")
                    # 重置状态，把当前内容当作新的起点
                    ctx.reset_for_new_target()
                    peak_text_len = current_text_len
                    content_shrink_count = 0
                    silence_start = time.time()
                    last_output_time = None
                    last_text_len = current_text_len
                    # 不要继续处理这一轮，等下一轮
                    time.sleep(0.2)
                    continue
            else:
                content_shrink_count = 0
        
            # ===== v4.1：检测是否出现新节点（思考 -> 最终答复切换）=====
            if current_count > ctx.output_target_count:
                if current_anchor != ctx.output_target_anchor:
                    if ctx.pending_new_anchor == current_anchor:
                        ctx.pending_new_anchor_seen += 1
                    else:
                        ctx.pending_new_anchor = current_anchor
                        ctx.pending_new_anchor_seen = 1
                
                    if ctx.pending_new_anchor_seen >= 2:
                        logger.info(f"[Switch] 检测到新节点出现，切换输出目标: "
                                   f"count {ctx.output_target_count} -> {current_count}, "
                                   f"anchor {ctx.output_target_anchor} -> {current_anchor}")
                    
                        ctx.output_target_count = current_count
                        ctx.output_target_anchor = current_anchor
                        ctx.pending_new_anchor = None
                        ctx.pending_new_anchor_seen = 0
                    
                        ctx.reset_for_new_target()
                        peak_text_len = 0  # v4.2：重置峰值
                        silence_start = time.time()
                        last_output_time = None
                        last_text_len = 0
                    
                        if not current_text:
                            time.sleep(0.2)
                            continue
            else:
                ctx.pending_new_anchor = None
                ctx.pending_new_anchor_seen = 0
        
            # ===== 处理空文本 =====
            if not current_text:
                if ctx.sent_content_length > 0:
                    element_missing_count += 1
                    if element_missing_count >= max_element_missing:
                        logger.warning("元素持续丢失，退出监控")
                        break
                time.sleep(0.2)
                continue
            else:
                element_missing_count = 0
        
            # 更新最大文本
            if len(current_text) > len(ctx.max_seen_text):
                ctx.max_seen_text = current_text
        
            # 计算增量
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
            
                ctx.content_ever_changed = True
            
                logger.debug(f"[Output] 发送增量: {len(diff)} 字符")
                yield self.formatter.pack_chunk(diff)
            else:
                if current_text == ctx.last_stable_text:
                    ctx.stable_text_count += 1
                else:
                    ctx.stable_text_count = 0
                    ctx.last_stable_text = current_text
            
                current_interval = min(current_interval * 1.5, max_interval)
        
            # 检测内容长度变化
            if current_text_len != last_text_len:
                ctx.content_ever_changed = True
                last_text_len = current_text_len
        
            silence_duration = time.time() - silence_start
        
            # ===== 退出判定 =====
            if ctx.content_ever_changed:
                if not still_generating:
                    if silence_duration > BrowserConstants.STREAM_SILENCE_THRESHOLD:
                        logger.info(f"[Exit] 生成结束（指示器消失 + {silence_duration:.1f}s 静默）")
                        break
                else:
                    if (ctx.stable_text_count >= BrowserConstants.STREAM_STABLE_COUNT_THRESHOLD and
                        silence_duration > BrowserConstants.STREAM_SILENCE_THRESHOLD_FALLBACK):
                        logger.info(f"[Exit] 生成结束（内容稳定 + {silence_duration:.1f}s 静默）")
                        break
            else:
                if not still_generating and last_output_time is None:
                    if current_text_len > ctx.active_turn_baseline_len + 5:
                        logger.info("[Exit] 检测到快速回复（无增量但有最终内容）")
                        break
        
            # 分段 sleep
            sleep_elapsed = 0
            while sleep_elapsed < current_interval:
                if self._check_cancelled():
                    break
                time.sleep(min(0.1, current_interval - sleep_elapsed))
                sleep_elapsed += 0.1
    
        # v4.1：最终读取（带 settle）
        if not self._check_cancelled():
            yield from self._final_settle_and_output(selector, ctx)


    def _final_settle_and_output(self, selector: str, ctx: StreamContext) -> Generator[str, None, None]:
        """
        最终稳定读取（v4.1 新增）
    
        在生成结束后等待一小段时间，确保：
        1. 如果有新节点出现（第三节点），切换并输出它
        2. 最后节点的内容已经完全渲染
        """
        settle_time = 1.5  # settle 窗口
        hardcap = 5.0      # 硬上限
    
        start = time.time()
        stable_start = time.time()
        last_snap = self._get_last_message_snapshot(selector)
    
        logger.debug(f"[Settle] 开始最终稳定等待，当前 count={last_snap['groups_count']}")
    
        while True:
            if self._check_cancelled():
                break
        
            now = time.time()
            if now - start > hardcap:
                break
        
            if now - stable_start >= settle_time:
                break
        
            time.sleep(0.15)
            snap = self._get_last_message_snapshot(selector)
        
            # 检测是否有新节点或内容变化
            changed = False
        
            if snap['groups_count'] > last_snap['groups_count']:
                logger.info(f"[Settle] 检测到新节点: count {last_snap['groups_count']} -> {snap['groups_count']}")
                changed = True
                # 新节点出现，切换目标
                if snap['anchor'] != ctx.output_target_anchor:
                    logger.info(f"[Settle] 切换到新节点: {snap['anchor']}")
                    ctx.output_target_anchor = snap['anchor']
                    ctx.output_target_count = snap['groups_count']
                    ctx.reset_for_new_target()
        
            if snap['text_len'] != last_snap['text_len']:
                changed = True
        
            if snap['anchor'] != last_snap['anchor']:
                changed = True
        
            if changed:
                stable_start = time.time()
        
            last_snap = snap
    
        # 读取最终内容
        final_snap = self._get_last_message_snapshot(selector)
        final_text = final_snap.get('text', "") or ""
    
        logger.debug(f"[Settle] 最终快照: count={final_snap['groups_count']}, text_len={len(final_text)}")
    
        # 计算未发送的剩余内容
        if final_text:
            final_effective_start = ctx.active_turn_baseline_len + ctx.sent_content_length
            if len(final_text) > final_effective_start:
                remaining = final_text[final_effective_start:]
                if remaining.strip():
                    logger.debug(f"[Final] 发送剩余内容: {len(remaining)} 字符")
                    yield self.formatter.pack_chunk(remaining)
                    ctx.sent_content_length += len(remaining)
        
            self._final_complete_text = final_text[ctx.active_turn_baseline_len:]
        else:
            # 如果最后节点为空，尝试回退到最后一个非空节点
            fallback_text = self._get_active_turn_text(selector)
            if fallback_text:
                final_effective_start = ctx.active_turn_baseline_len + ctx.sent_content_length
                if len(fallback_text) > final_effective_start:
                    remaining = fallback_text[final_effective_start:]
                    if remaining.strip():
                        logger.debug(f"[Final Fallback] 发送剩余内容: {len(remaining)} 字符")
                        yield self.formatter.pack_chunk(remaining)
                        ctx.sent_content_length += len(remaining)
            
                self._final_complete_text = fallback_text[ctx.active_turn_baseline_len:]
            else:
                self._final_complete_text = ctx.max_seen_text[ctx.active_turn_baseline_len:] if ctx.max_seen_text else ""
    
        logger.info(f"========== 流式监听结束，总输出: {ctx.sent_content_length} 字符 ==========")


    def _get_last_message_snapshot(self, selector: str) -> dict:
        """
        获取最后一个消息节点的快照（v4.1 新增）
    
        与 _get_latest_message_snapshot 不同：这里永远取 eles[-1]，即使它是空的
        """
        result = {
            'groups_count': 0,
            'anchor': None,
            'text': '',
            'text_len': 0,
            'is_generating': False,
        }
    
        try:
            eles = self.finder.find_all(selector, timeout=0.5)
        
            if not eles:
                return result
        
            result['groups_count'] = len(eles)
        
            last_ele = eles[-1]
            result['text'] = self._read_visible_text(last_ele) or ""
            result['text_len'] = len(result['text'])
            result['anchor'] = self._get_message_anchor(last_ele)
        
            if self._generating_checker is None:
                self._generating_checker = GeneratingStatusCache(self.tab)
            result['is_generating'] = self._generating_checker.is_generating()
    
        except Exception as e:
            logger.debug(f"Last snapshot 异常: {e}")
    
        return result


# ================= 浏览器核心 =================

class BrowserCore:
    """浏览器核心类 - 单例模式"""
    
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
        """设置停止检查器"""
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
        """工作流执行入口"""
        
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
        
        # 创建执行器
        executor = WorkflowExecutor(
            tab, 
            self.stealth_mode,
            should_stop_checker=self._should_stop_checker
        )
        
        try:
            for step in workflow:
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
