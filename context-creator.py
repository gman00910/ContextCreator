# New optimized context creator!!!
import glob
import os
import re
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Set, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from collections import counter, defaultdict as dd

# Use dd as an alias for defaultdict
defaultdict = dd
import asyncio
import yaml
import json
import logging
import aiofiles 

import ast
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go     #*** We can take this stuff out bc I told AI I only waatned terminal visuals, so we went with rich....RIGHT????????
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

import networkx as nx
import rich
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text
from rich.syntax import Syntax


from .file_summary_utility import FileSummaryUtility
from .dependency_tracker import DependencyTracker
from .log_analysis_dashboard import LogAnalysisDashboard
from .enhanced_directory_structure import EnhancedDirectoryStructure



class ContextErrorType(Enum):
    """Classification of context creation errors"""
    CONFIG = "configuration"
    FILE_SYSTEM = "file_system"
    VALIDATION = "validation"
    PARSING = "parsing"
    PERMISSION = "permission"
    ENCODING = "encoding"
    MEMORY = "memory"
    NETWORK = "network"
    UNKNOWN = "unknown"

@dataclass
class ContextError:
    """Structured error information"""
    error_type: ContextErrorType
    message: str
    details: Optional[str] = None
    traceback: Optional[str] = None
    recovery_hint: Optional[str] = None
    component: Optional[str] = None

class ErrorHandler:
    """Centralized error handling with recovery suggestions"""
    
    def __init__(self, event_emitter: Optional['ContextEventEmitter'] = None):
        self.event_emitter = event_emitter
        self.error_counts: Dict[ContextErrorType, int] = {
            error_type: 0 for error_type in ContextErrorType
        }
        
        # Recovery hints for different error types
        self.recovery_hints = {
            ContextErrorType.CONFIG: "Check configuration file syntax and required fields",
            ContextErrorType.FILE_SYSTEM: "Verify file permissions and paths",
            ContextErrorType.VALIDATION: "Review input data format and requirements",
            ContextErrorType.PARSING: "Check file format and encoding",
            ContextErrorType.PERMISSION: "Verify user permissions for accessed resources",
            ContextErrorType.ENCODING: "Ensure correct file encoding (UTF-8 recommended)",
            ContextErrorType.MEMORY: "Check available system resources",
            ContextErrorType.NETWORK: "Verify network connectivity and permissions"
        }
    
    async def handle_error(self, 
                         error: Exception, 
                         component: str = None,
                         error_type: ContextErrorType = ContextErrorType.UNKNOWN) -> ContextError:
        """Handle and classify error with async support"""
        try:
            # Determine error type if not specified
            if error_type == ContextErrorType.UNKNOWN:
                error_type = self._classify_error(error)
            
            # Build error details
            error_info = ContextError(
                error_type=error_type,
                message=str(error),
                details=self._get_error_details(error),
                traceback=traceback.format_exc(),
                recovery_hint=self.recovery_hints.get(error_type),
                component=component
            )
            
            # Update error count
            self.error_counts[error_type] += 1
            
            # Emit error event if event emitter is available
            if self.event_emitter:
                await self.event_emitter.emit("error", {
                    "type": error_type.value,
                    "message": str(error),
                    "component": component
                })
            
            return error_info
            
        except Exception as e:
            # Fallback error handling
            return ContextError(
                error_type=ContextErrorType.UNKNOWN,
                message=f"Error handling failed: {str(e)}",
                traceback=traceback.format_exc()
            )
    
    def _classify_error(self, error: Exception) -> ContextErrorType:
        """Classify error type based on exception"""
        error_class = error.__class__.__name__
        
        error_map = {
            'FileNotFoundError': ContextErrorType.FILE_SYSTEM,
            'PermissionError': ContextErrorType.PERMISSION,
            'UnicodeError': ContextErrorType.ENCODING,
            'UnicodeDecodeError': ContextErrorType.ENCODING,
            'UnicodeEncodeError': ContextErrorType.ENCODING,
            'MemoryError': ContextErrorType.MEMORY,
            'ValueError': ContextErrorType.VALIDATION,
            'JSONDecodeError': ContextErrorType.PARSING,
            'YAMLError': ContextErrorType.PARSING,
            'ConnectionError': ContextErrorType.NETWORK,
            'TimeoutError': ContextErrorType.NETWORK,
        }
        
        return error_map.get(error_class, ContextErrorType.UNKNOWN)
    
    def _get_error_details(self, error: Exception) -> Optional[str]:
        """Extract detailed error information"""
        try:
            if hasattr(error, 'strerror'):
                return error.strerror
            elif hasattr(error, '__context__') and error.__context__:
                return str(error.__context__)
            return None
        except:
            return None
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of encountered errors"""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_counts': {
                error_type.value: count
                for error_type, count in self.error_counts.items()
                if count > 0
            }
        }

class FileValidator:
    """Validate file operations and content"""
    
    @staticmethod
    async def validate_file_access(path: str, required_perms: set = {'read'}) -> tuple[bool, Optional[str]]:
        """Validate file accessibility with specific permissions"""
        try:
            file_path = Path(path)
            
            # Check existence
            if not file_path.exists():
                return False, f"File not found: {path}"
            
            # Check permissions
            if 'read' in required_perms and not os.access(path, os.R_OK):
                return False, f"Read permission denied: {path}"
            if 'write' in required_perms and not os.access(path, os.W_OK):
                return False, f"Write permission denied: {path}"
            
            return True, None
            
        except Exception as e:
            return False, f"Error checking file access: {str(e)}"
    
    @staticmethod
    async def validate_file_type(path: str, allowed_types: set) -> tuple[bool, Optional[str]]:
        """Validate file type and basic structure"""
        try:
            file_path = Path(path)
            
            # Check file extension
            ext = file_path.suffix.lower()
            if ext not in allowed_types:
                return False, f"Invalid file type: {ext}. Allowed: {allowed_types}"
            
            # Additional type-specific validation
            if ext == '.py':
                valid, msg = await FileValidator._validate_python_file(file_path)
            elif ext in {'.yml', '.yaml'}:
                valid, msg = await FileValidator._validate_yaml_file(file_path)
            else:
                valid, msg = True, None
            
            return valid, msg
            
        except Exception as e:
            return False, f"Error validating file type: {str(e)}"
    
    @staticmethod
    async def _validate_python_file(path: Path) -> tuple[bool, Optional[str]]:
        """Validate Python file syntax"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            compile(content, path, 'exec')
            return True, None
        except SyntaxError as e:
            return False, f"Python syntax error: {str(e)}"
        except Exception as e:
            return False, f"Error validating Python file: {str(e)}"
    
    @staticmethod
    async def _validate_yaml_file(path: Path) -> tuple[bool, Optional[str]]:
        """Validate YAML file structure"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                yaml.safe_load(f)
            return True, None
        except yaml.YAMLError as e:
            return False, f"YAML parsing error: {str(e)}"
        except Exception as e:
            return False, f"Error validating YAML file: {str(e)}"


class ConfigValidator:
    """Validation for configuration files and settings"""
    
    @staticmethod
    async def validate_config_file(config_path: str) -> tuple[bool, Optional[str]]:
        """Validate config file structure and required fields"""
        try:
            if not os.path.exists(config_path):
                return False, f"Config file not found: {config_path}"
            
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            # Required sections
            required = {'core_files', 'path_mappings'}
            missing = required - set(config.keys())
            if missing:
                return False, f"Missing required sections: {missing}"
            
            # Validate path mappings
            if not isinstance(config['path_mappings'], dict):
                return False, "path_mappings must be a dictionary"
            
            for component, paths in config['path_mappings'].items():
                if not isinstance(paths, list):
                    return False, f"Invalid paths for component {component}"
            
            return True, None
            
        except yaml.YAMLError as e:
            return False, f"YAML parsing error: {str(e)}"
        except Exception as e:
            return False, f"Config validation error: {str(e)}"


class LogValidator:
    """Validation for log files and entries"""
    
    @staticmethod
    async def validate_log_file(log_path: str) -> tuple[bool, Optional[str]]:
        """Validate log file format and accessibility"""
        try:
            if not os.path.exists(log_path):
                return False, f"Log file not found: {log_path}"
            
            if not os.access(log_path, os.R_OK):
                return False, f"Log file not readable: {log_path}"
            
            # Check first few lines for expected format
            with open(log_path, 'r') as f:
                first_lines = [next(f) for _ in range(3)]
                
            # Basic timestamp validation
            timestamp_pattern = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
            if not any(timestamp_pattern.search(line) for line in first_lines):
                return False, "Log file appears to have invalid format"
            
            return True, None
            
        except Exception as e:
            return False, f"Log validation error: {str(e)}"

class RetryHandler:
    """Handle retries for recoverable errors"""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay
        self.retry_counts: Dict[str, int] = {}
    
    async def execute_with_retry(self, 
                               func: Callable, 
                               *args,
                               operation_name: str = "unknown",
                               **kwargs) -> Any:
        """Execute function with retry logic"""
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                result = await func(*args, **kwargs)
                self.retry_counts[operation_name] = retries
                return result
            
            except (IOError, OSError) as e:
                # File system errors might be recoverable
                retries += 1
                last_error = e
                if retries < self.max_retries:
                    await asyncio.sleep(self.delay * retries)
                continue
                
            except Exception as e:
                # Don't retry other errors
                raise
        
        # Max retries reached
        self.retry_counts[operation_name] = self.max_retries
        raise last_error or Exception(f"Max retries reached for {operation_name}")
    
    def get_retry_stats(self) -> Dict[str, int]:
        """Get retry statistics by operation"""
        return dict(self.retry_counts)

class ContextValidator:
    """Validate context operations and content"""
    
    @staticmethod
    def validate_context_type(context_type: str, valid_types: Set[str]) -> tuple[bool, Optional[str]]:
        """Validate context type selection"""
        if not context_type:
            return False, "Context type cannot be empty"
        
        if context_type not in valid_types:
            return False, f"Invalid context type. Valid types: {valid_types}"
        
        return True, None
    
    @staticmethod
    def validate_content(content: str) -> tuple[bool, Optional[str]]:
        """Validate context content"""
        if not content:
            return False, "Context content cannot be empty"
        
        # Check for minimum content
        if len(content.strip()) < 50:
            return False, "Context content too short"
        
        # Check for maximum size (e.g., 10MB)
        if len(content.encode('utf-8')) > 10_000_000:
            return False, "Context content too large"
        
        return True, None

# Custom Exceptions
class ConfigValidationError(Exception):
    """Configuration validation errors"""
    pass

class FileAccessError(Exception):
    """File access errors"""
    pass

class EncodingError(Exception):
    """Encoding/decoding errors"""
    pass

class ContextError(Exception):
    """Context creation/manipulation errors"""
    pass

# Enhanced Logging Setup
class ErrorLogger:
    """Centralized error logging with context tracking"""
    
    def __init__(self, log_file: str = "context_creator.log"):
        self.log_file = log_file
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging with appropriate formatting"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        
        # Add custom logging levels for context operations
        logging.addLevelName(25, "OPERATION")
        logging.addLevelName(15, "VALIDATION")
    
    def log_error(self, error: Exception, context: dict = None):
        """Log error with context information"""
        error_info = {
            'error_type': error.__class__.__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        logging.error(
            "Error occurred: %(error_type)s - %(message)s\nContext: %(context)s",
            error_info
        )
    
    def log_operation(self, operation: str, details: dict = None):
        """Log operation with details"""
        logging.log(
            25,  # OPERATION level
            f"Operation: {operation} - Details: {details or {}}"
        )
    
    def log_validation(self, component: str, result: bool, message: str = None):
        """Log validation results"""
        logging.log(
            15,  # VALIDATION level
            f"Validation [{component}] - {'Success' if result else 'Failed'}: {message or ''}"
        )

# Error Handler Integration
class IntegratedErrorHandler:
    """Integrated error handling for context operations"""
    
    def __init__(self, logger: ErrorLogger):
        self.logger = logger
        self.retry_handler = RetryHandler()
        self.error_counts: Dict[str, int] = defaultdict(int)
    
    async def handle_operation(self, 
                             operation: Callable, 
                             *args,
                             retry: bool = True,
                             operation_name: str = None,
                             **kwargs) -> Any:
        """Handle operation with integrated error handling"""
        try:
            if retry:
                result = await self.retry_handler.execute_with_retry(
                    operation,
                    *args,
                    operation_name=operation_name or operation.__name__,
                    **kwargs
                )
            else:
                result = await operation(*args, **kwargs)
            
            self.logger.log_operation(
                operation_name or operation.__name__,
                {'status': 'success', 'args': args, 'kwargs': kwargs}
            )
            
            return result
            
        except ConfigValidationError as e:
            self.error_counts['config'] += 1
            self.logger.log_error(e, {
                'operation': operation_name,
                'config_related': True
            })
            raise
            
        except FileAccessError as e:
            self.error_counts['file_access'] += 1
            self.logger.log_error(e, {
                'operation': operation_name,
                'file_related': True
            })
            raise
            
        except EncodingError as e:
            self.error_counts['encoding'] += 1
            self.logger.log_error(e, {
                'operation': operation_name,
                'encoding_related': True
            })
            raise
            
        except Exception as e:
            self.error_counts['unknown'] += 1
            self.logger.log_error(e, {
                'operation': operation_name,
                'unexpected': True
            })
            raise









@dataclass
class ConfigSettings:
    """Enhanced configuration settings with validation and serialization"""
    core_files: Dict[str, List[str]]
    exclude_patterns: List[str]
    path_mappings: Dict[str, List[str]]
    component_relationships: Dict[str, Dict]
    system_flows: Dict[str, Dict]
    
    def to_dict(self) -> dict:
        """Convert settings to dictionary"""
        return asdict(self)
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        try:
            # Check required sections
            required_sections = ['core_files', 'path_mappings']
            for section in required_sections:
                if not getattr(self, section):
                    raise ValueError(f"Missing required section: {section}")
            
            # Validate path mappings
            for component, paths in self.path_mappings.items():
                if not isinstance(paths, list):
                    raise ValueError(f"Invalid paths for component {component}")
                
            return True
        except Exception as e:
            logging.error(f"Configuration validation error: {str(e)}")
            return False
        




class ConfigLoader:
    """Base configuration loader class"""
    
    def __init__(self):
        self._cache = {}
        self._lock = asyncio.Lock()
    
    async def load_config(self, config_path: str) -> ConfigSettings:
        """Load configuration from file"""
        try:
            config_data = self._read_yaml_file(config_path)
            return ConfigSettings(**config_data)
        except Exception as e:
            raise ConfigValidationError(f"Error loading config: {str(e)}")
    
    @staticmethod
    def _read_yaml_file(path: str) -> dict:
        """Read and parse YAML file"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)




# Integration with ConfigLoader (w error handling stuff up top)
class EnhancedConfigLoader(ConfigLoader):
    """ConfigLoader with integrated error handling"""
    
    def __init__(self):
        super().__init__()
        self.logger = ErrorLogger()
        self.error_handler = IntegratedErrorHandler(self.logger)
        self.validator = ConfigValidator()
    

    
    async def get_cached_config(self, config_path: str, max_age: float = 300) -> ConfigSettings:
        """Get configuration with caching"""
        async with self._lock:
            now = datetime.now().timestamp()
            if config_path in self._cache:
                config, timestamp = self._cache[config_path]
                if now - timestamp < max_age:
                    return config
            
            config = await self.load_config(config_path)
            self._cache[config_path] = (config, now)
            return config
        






class FileSystemCache:
    """Enhanced file system cache with improved error handling"""
    
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._content_cache: Dict[str, tuple[str, float]] = {}
        self._mtime_cache: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    @lru_cache(maxsize=1000)
    async def get_file_content(self, file_path: str, check_mtime: bool = True) -> str:
        """Get file content with modification time checking"""
        try:
            async with self._lock:
                if check_mtime and await self._file_modified(file_path):
                    self.get_file_content.cache_clear()
                
                loop = asyncio.get_event_loop()
                content = await loop.run_in_executor(
                    self.executor,
                    self._read_file,
                    file_path
                )
                return content
                
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {str(e)}")
            return f"Error reading file: {str(e)}"
    
    async def _file_modified(self, file_path: str) -> bool:
        """Check if file has been modified"""
        try:
            current_mtime = Path(file_path).stat().st_mtime
            last_mtime = self._mtime_cache.get(file_path, 0)
            
            if current_mtime > last_mtime:
                self._mtime_cache[file_path] = current_mtime
                return True
            return False
            
        except Exception:
            return True
    
    def _read_file(self, file_path: str) -> str:
        """Synchronous file reading with encoding handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback to binary reading if UTF-8 fails
            with open(file_path, 'rb') as f:
                return f.read().decode('utf-8', errors='replace')





class ContextEventEmitter:
    """Enhanced event system with typed events and async support"""
    
    def __init__(self):
        self.subscribers: List[Callable[[str, dict], None]] = []
        self.async_subscribers: List[Callable[[str, dict], asyncio.coroutine]] = []
        self._lock = asyncio.Lock()
        self._event_history: List[dict] = []
    
    def subscribe(self, callback: Callable[[str, dict], None]):
        """Subscribe to synchronous events"""
        self.subscribers.append(callback)
    
    def subscribe_async(self, callback: Callable[[str, dict], asyncio.coroutine]):
        """Subscribe to async events"""
        self.async_subscribers.append(callback)
    
    async def emit(self, event_type: str, data: dict):
        """Emit event with async support"""
        async with self._lock:
            # Record event
            event_record = {
                'timestamp': datetime.now().isoformat(),
                'type': event_type,
                'data': data
            }
            self._event_history.append(event_record)
            
            # Call sync subscribers
            for subscriber in self.subscribers:
                try:
                    subscriber(event_type, data)
                except Exception as e:
                    logging.error(f"Error in event subscriber: {str(e)}")
            
            # Call async subscribers
            tasks = [
                subscriber(event_type, data)
                for subscriber in self.async_subscribers
            ]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_recent_events(self, limit: int = 100) -> List[dict]:
        """Get recent events from history"""
        return self._event_history[-limit:]
    
    def clear_history(self):
        """Clear event history"""
        self._event_history.clear()





class ContextManager:
    """
    High-level manager class to coordinate context creation operations
    and maintain overall system state.
    """
    
    def __init__(self, base_dir: str = ".", config_path: str = "context_config.yaml"):
        self.creator = EnhancedContextCreator(base_dir, config_path)
        self.current_context = None
        self.last_operation = None
        self.operation_history = []
        
    async def initialize(self):
        """Initialize the context manager and verify system state"""
        try:
            # Subscribe to context creator events
            self.creator.events.subscribe(self._handle_event)
            
            # Verify directory structure
            tree = await self.creator._build_directory_structure()
            if not tree:
                raise ValueError("Invalid directory structure")
            
            # Initialize stat collection
            stats = await self.creator._calculate_project_stats()
            self.creator.python_files = stats['files']
            self.creator.total_lines = stats['lines']
            
            return True
            
        except Exception as e:
            print(f"\033[91mInitialization error: {str(e)}\033[0m")
            return False
    
    def _handle_event(self, event_type: str, data: dict):
        """Handle events from context creator"""
        # Log the event
        self.operation_history.append({
            'timestamp': datetime.now(),
            'event': event_type,
            'data': data
        })
        
        # Handle specific events
        if event_type == "context_complete":
            self.last_operation = {
                'type': 'context_creation',
                'status': 'complete',
                'size': data.get('size', 0)
            }
        elif event_type == "context_error":
            self.last_operation = {
                'type': 'context_creation',
                'status': 'error',
                'error': data.get('error')
            }
    
    async def create_context(self, context_type: str, include_logs: bool = False):
        """Create context with progress tracking and state management"""
        try:
            self.current_context = await self.creator.create_context(
                context_type,
                include_logs=include_logs
            )
            return self.current_context
        except Exception as e:
            print(f"\033[91mError creating context: {str(e)}\033[0m")
            return None
    
    async def export_context(self, filename: str = None):
        """Export current context to file with proper formatting"""
        if not self.current_context:
            print("\033[93mNo context available to export\033[0m")
            return False
            
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"context_{timestamp}.md"
            
            await self.creator._save_context(self.current_context, filename)
            return True
            
        except Exception as e:
            print(f"\033[91mError exporting context: {str(e)}\033[0m")
            return False
    
    async def get_operation_summary(self) -> dict:
        """Get summary of recent operations and system state"""
        return {
            'last_operation': self.last_operation,
            'history_length': len(self.operation_history),
            'current_context_size': len(self.current_context) if self.current_context else 0,
            'python_files': self.creator.python_files,
            'total_lines': self.creator.total_lines
        }










class DependencyTracker:
    """
    Advanced dependency tracking for Python projects
    
    Key Features:
    - Analyze import statements (direct and relative)
    - Create dependency graphs
    - Identify circular dependencies
    - Generate visualization
    """
    
    def __init__(self, base_dir: str = "."):
        """
        Initialize dependency tracker
        
        Args:
            base_dir: Root directory of the project
        """
        self.base_dir = Path(base_dir)
        self.dependency_graph = nx.DiGraph()
        self.module_to_file_map: Dict[str, str] = {}
        
        # Exclude patterns for files/directories
        self.exclude_patterns = [
            '*__pycache__*', 
            '*.pyc', 
            '*/.git/*', 
            '*.log', 
            '*/tests/*', 
            '*/venv/*'
        ]
    
    def _is_excluded(self, path: Path) -> bool:
        """
        Check if a file should be excluded from analysis
        
        Args:
            path: File path to check
        
        Returns:
            Boolean indicating if file should be excluded
        """
        return any(
            re.search(pattern.replace('*', '.*'), str(path)) 
            for pattern in self.exclude_patterns
        )
    
    def _extract_imports(self, file_path: Path) -> List[Tuple[str, str]]:
        """
        Extract import statements from a Python file
        
        Args:
            file_path: Path to the Python file
        
        Returns:
            List of tuples (import_type, module_name)
        """
        imports = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                # Standard imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(('import', alias.name))
                
                # From ... import statements
                elif isinstance(node, ast.ImportFrom):
                    # Determine module name, handling relative imports
                    if node.module:
                        module_name = node.module
                        # Convert relative imports
                        if node.level > 0:
                            # Get the current module's path
                            current_module = file_path.relative_to(self.base_dir).with_suffix('')
                            module_parts = str(current_module).split(os.path.sep)
                            
                            # Handle relative import levels
                            relative_parts = module_parts[:-node.level]
                            if module_name:
                                relative_parts.append(module_name)
                            
                            module_name = '.'.join(relative_parts)
                        
                        imports.append(('from', module_name))
                    
                    # Import specific names
                    for alias in node.names:
                        imports.append(('imported_name', alias.name))
            
            return imports
        
        except SyntaxError:
            print(f"Syntax error in {file_path}")
            return []
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []
    
    def build_dependency_map(self) -> Dict[str, Set[str]]:
        """
        Build a comprehensive dependency map for the entire project
        
        Returns:
            Dictionary mapping file paths to their dependencies
        """
        dependency_map: Dict[str, Set[str]] = {}
        
        # Walk through the project directory
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                file_path = Path(root) / file
                
                # Skip excluded files and non-Python files
                if self._is_excluded(file_path) or not file_path.suffix == '.py':
                    continue
                
                try:
                    # Get imports for this file
                    imports = self._extract_imports(file_path)
                    
                    # Convert file path to module name
                    relative_path = file_path.relative_to(self.base_dir)
                    module_name = str(relative_path.with_suffix('')).replace(os.path.sep, '.')
                    
                    # Store module to file mapping
                    self.module_to_file_map[module_name] = str(file_path)
                    
                    # Track dependencies
                    dependency_set = set()
                    for import_type, module in imports:
                        if import_type in ['import', 'from']:
                            dependency_set.add(module)
                    
                    dependency_map[str(file_path)] = dependency_set
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return dependency_map
    
    def generate_dependency_graph(self) -> nx.DiGraph:
        """
        Generate a directed dependency graph
        
        Returns:
            NetworkX directed graph of project dependencies
        """
        # Build dependency map first
        dependency_map = self.build_dependency_map()
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for file, dependencies in dependency_map.items():
            G.add_node(file)
            for dep in dependencies:
                # Try to resolve dependency to a file path
                if dep in self.module_to_file_map:
                    G.add_edge(file, self.module_to_file_map[dep])
        
        return G
    
    def visualize_dependency_graph(self, output_path: str = 'dependency_graph.png'):
        """
        Visualize the dependency graph
        
        Args:
            output_path: Path to save the graph visualization
        """
        # Generate the graph
        G = self.generate_dependency_graph()
        
        # Set up the plot
        plt.figure(figsize=(20, 20))
        
        # Use spring layout for graph positioning
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Draw the graph
        nx.draw(
            G, 
            pos, 
            with_labels=True, 
            node_color='lightblue', 
            node_size=300, 
            font_size=8, 
            font_weight='bold',
            arrows=True,
            edge_color='gray'
        )
        
        plt.title("Project Dependency Graph")
        plt.tight_layout()
        
        # Save the graph
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_circular_dependencies(self) -> List[List[str]]:
        """
        Detect and return circular dependencies
        
        Returns:
            List of circular dependency cycles
        """
        G = self.generate_dependency_graph()
        
        # Find all simple cycles
        cycles = list(nx.simple_cycles(G))
        
        return cycles
    
    def generate_dependency_report(self) -> str:
        """
        Generate a comprehensive dependency report
        
        Returns:
            Markdown-formatted dependency report
        """
        report = ["# Project Dependency Analysis\n"]
        
        # Dependency Map
        report.append("## Dependency Map\n")
        dependency_map = self.build_dependency_map()
        for file, dependencies in dependency_map.items():
            report.append(f"### {file}")
            if dependencies:
                report.append("Dependencies:")
                for dep in dependencies:
                    report.append(f"- {dep}")
            else:
                report.append("- No external dependencies")
            report.append("")
        
        # Circular Dependencies
        report.append("## Circular Dependencies\n")
        circular_deps = self.analyze_circular_dependencies()
        if circular_deps:
            for cycle in circular_deps:
                report.append("- " + " → ".join(cycle))
        else:
            report.append("No circular dependencies detected.")
        
        return "\n".join(report)










class FileLineNumberer:
    """
    Advanced file line numbering and section extraction utility
    """
    
    @staticmethod
    def add_line_numbers(content: str, start_line: int = 1) -> str:
        """
        Add line numbers to file content
        
        Args:
            content: File content as a string
            start_line: Starting line number (default: 1)
        
        Returns:
            Content with line numbers
        """
        lines = content.splitlines()
        numbered_lines = [
            f"{start_line + i:4d} | {line}" 
            for i, line in enumerate(lines)
        ]
        return "\n".join(numbered_lines)
    
    @staticmethod
    def get_line_info(file_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive information about file lines
        
        Args:
            file_path: Path to the file
        
        Returns:
            Dictionary with file line information
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            return {
                'path': file_path,
                'total_lines': len(lines),
                'lines': lines,
                'line_numbers': list(range(1, len(lines) + 1))
            }
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return {}
    
    @staticmethod
    def find_line_numbers(file_path: str, search_term: Optional[str] = None) -> List[int]:
        """
        Find line numbers matching a search term
        
        Args:
            file_path: Path to the file
            search_term: Optional term to search for
        
        Returns:
            List of line numbers matching the search term
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            if search_term:
                return [
                    i + 1 for i, line in enumerate(lines)
                    if search_term.lower() in line.lower()
                ]
            
            return list(range(1, len(lines) + 1))
        
        except Exception as e:
            print(f"Error finding line numbers in {file_path}: {e}")
            return []
    
    @staticmethod
    def extract_file_section(
        file_path: str, 
        start_line: Optional[int] = None, 
        end_line: Optional[int] = None
    ) -> str:
        """
        Extract a specific section of a file
        
        Args:
            file_path: Path to the file
            start_line: Starting line number (inclusive)
            end_line: Ending line number (inclusive)
        
        Returns:
            Extracted file section with line numbers
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Adjust line numbers to 0-based indexing
            start = max(0, (start_line or 1) - 1)
            end = min(len(lines), (end_line or len(lines)))
            
            # Extract section with original line numbers
            section_lines = [
                f"{start + i + 1:4d} | {line}" 
                for i, line in enumerate(lines[start:end])
            ]
            
            return "\n".join(section_lines)
        
        except Exception as e:
            print(f"Error extracting file section from {file_path}: {e}")
            return ""









class TerminalLogDashboard:
    def __init__(self, log_directory='logs'):
        self.log_directory = log_directory
        self.console = Console()  # Using rich for terminal visualization
    
    def generate_log_summary(self):
        """
        Generate a comprehensive log summary with rich terminal formatting
        """
        # Collect and parse logs
        log_files = self._collect_log_files()
        
        # Create rich panel for log summary
        summary_panel = Panel(
            self._analyze_logs(log_files),
            title="[bold blue]🔍 Log Analysis Summary[/bold blue]",
            border_style="blue"
        )
        
        # Print to terminal
        self.console.print(summary_panel)
        
        # Optional: Write to separate log file
        self._write_log_summary_file(log_files)
    
    def _collect_log_files(self):
        """Collect and filter log files"""
        return [
            f for f in os.listdir(self.log_directory) 
            if f.endswith('.log')
        ]
    
    def _analyze_logs(self, log_files):
        """Analyze log files with rich formatting"""
        log_analysis = []
        
        for log_file in log_files:
            full_path = os.path.join(self.log_directory, log_file)
            
            # Analyze log file contents
            log_content = self._parse_log_file(full_path)
            
            log_analysis.append(
                Text.assemble(
                    (f"\n📄 {log_file}:\n", "bold green"),
                    *log_content
                )
            )
        
        return log_analysis
    
    def _parse_log_file(self, log_path):
        """Parse individual log file"""
        log_entries = []
        
        with open(log_path, 'r') as f:
            for line in f:
                # Color-code log levels
                if 'ERROR' in line:
                    log_entries.append((line.strip(), "bold red"))
                elif 'WARNING' in line:
                    log_entries.append((line.strip(), "yellow"))
                elif 'INFO' in line:
                    log_entries.append((line.strip(), "green"))
                else:
                    log_entries.append((line.strip(), "white"))
        
        return log_entries
    
    def _write_log_summary_file(self, log_files):
        """Write log summary to a separate file"""
        summary_path = 'log_summary.txt'
        with open(summary_path, 'w') as f:
            for log_file in log_files:
                full_path = os.path.join(self.log_directory, log_file)
                f.write(f"\n=== {log_file} ===\n")
                with open(full_path, 'r') as log:
                    f.write(log.read())
        
        print(f"\n💾 Detailed log summary saved to {summary_path}")



        






class EnhancedDirectoryStructure:
    """Enhanced directory structure with async capabilities and caching"""
    
    def __init__(self, root_path: str = ".", fs_cache: Optional[FileSystemCache] = None):
        self.root_path = Path(root_path)
        self.fs_cache = fs_cache or FileSystemCache()
        self.tree_cache = {}  # Cache for directory trees
        self.content_cache = {}  # Cache for file contents
        
    async def build_tree(self) -> Dict[str, Any]:
        """Async directory tree builder with caching"""
        cache_key = str(self.root_path)
        if cache_key in self.tree_cache:
            return self.tree_cache[cache_key]
            
        tree = {
            "src": {"files": []},
            "tests": {"files": []},
            "config": {"files": []},
            "scripts": {"files": []},
            "logs": {"files": []},
            "docs": {"files": []},
            "docker": {"files": []},
        }
        
        try:
            await self._scan_directory(self.root_path, tree)
            self.tree_cache[cache_key] = tree
            return tree
        except Exception as e:
            raise Exception(f"Error building directory tree: {str(e)}")
    
    async def _scan_directory(self, path: Path, structure: Dict[str, Any]):
        """Async directory scanner"""
        loop = asyncio.get_event_loop()
        paths = await loop.run_in_executor(
            self.fs_cache.executor,
            self._get_directory_contents,
            path
        )
        
        for item_path in paths:
            if '__pycache__' in str(item_path):
                continue
                
            await self._add_to_tree(item_path, structure)
    
    def _get_directory_contents(self, path: Path) -> List[Path]:
        """Synchronous directory content getter"""
        return list(path.glob('*'))
    
    async def _add_to_tree(self, path: Path, structure: Dict[str, Any]):
        """Async tree builder"""
        try:
            relative = path.relative_to(self.root_path)
            parts = relative.parts
            current = structure
            
            # Build path in tree
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {"files": []}
                current = current[part]
            
            # Add file or directory
            if path.is_file():
                current["files"].append(parts[-1])
            elif path.is_dir() and parts:
                if parts[-1] not in current:
                    current[parts[-1]] = {"files": []}
                    await self._scan_directory(path, current[parts[-1]])
                    
        except Exception as e:
            print(f"Error adding to tree: {str(e)}")
    
    async def display_structure(self, paths: List[str]):
        """Enhanced async structure display"""
        print("\n\033[96m📁 Project Structure:\033[0m")
        print("=" * 50)
        
        display_tasks = [
            self._display_path(path)
            for path in paths
        ]
        
        await asyncio.gather(*display_tasks)
    
    async def _display_path(self, path: str):
        """Async path display handler"""
        try:
            if not await self._path_exists(path):
                print(f"\033[91m❌ Path not found: {path}\033[0m")
                return
                
            if await self._is_directory(path):
                print(f"\n\033[94m📂 {path}/\033[0m")
                await self._display_directory_contents(path, prefix="  ")
            else:
                print(f"\033[37m📄 {path}\033[0m")
                await self._display_file_preview(path)
                
        except Exception as e:
            print(f"Error displaying path: {str(e)}")
    
    async def _display_directory_contents(self, path: str, prefix: str = "", level: int = 0):
        """Enhanced async directory content display"""
        if level > 3:  # Limit recursion depth
            print(f"{prefix}...")
            return
            
        try:
            items = await self._get_filtered_directory_items(path)
            
            for i, item in enumerate(sorted(items)):
                full_path = os.path.join(path, item)
                is_last = i == len(items) - 1
                
                if await self._is_directory(full_path):
                    print(f"{prefix}{'└── ' if is_last else '├── '}\033[94m📂 {item}/\033[0m")
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    await self._display_directory_contents(full_path, new_prefix, level + 1)
                else:
                    print(f"{prefix}{'└── ' if is_last else '├── '}\033[37m📄 {item}\033[0m")
                    
        except Exception as e:
            print(f"{prefix}\033[91mError reading directory: {str(e)}\033[0m")
    
    async def _get_filtered_directory_items(self, path: str) -> List[str]:
        """Get filtered directory items asynchronously"""
        loop = asyncio.get_event_loop()
        items = await loop.run_in_executor(
            self.fs_cache.executor,
            lambda: [
                item for item in os.listdir(path)
                if not any(glob.fnmatch.fnmatch(item, pat) 
                          for pat in self.fs_cache.config.exclude_patterns)
            ]
        )
        return items
    
    async def _display_file_preview(self, file_path: str, lines: int = 5):
        """Enhanced async file preview"""
        if not file_path.endswith(('.py', '.md', '.txt', '.yml', '.yaml', '.env')):
            return
            
        try:
            content = await self.fs_cache.get_file_content(file_path)
            preview_lines = content.splitlines()[:lines]
            
            print(f"    \033[90m{'=' * 40}\033[0m")
            for line in preview_lines:
                print(f"    \033[90m{line.rstrip()}\033[0m")
            
            if len(preview_lines) < content.count('\n'):
                print("    \033[90m...\033[0m")
                
            print(f"    \033[90m{'=' * 40}\033[0m")
            
        except Exception as e:
            print(f"    \033[91mError reading file: {str(e)}\033[0m")
    
    @staticmethod
    async def _path_exists(path: str) -> bool:
        """Async path existence check"""
        return Path(path).exists()
    
    @staticmethod
    async def _is_directory(path: str) -> bool:
        """Async directory check"""
        return Path(path).is_dir()
    
    async def get_project_stats(self) -> Dict[str, int]:
        """Get project statistics asynchronously"""
        stats = {
            'files': 0,
            'lines': 0
        }
        
        try:
            python_files = await self._find_python_files()
            stats['files'] = len(python_files)
            
            line_counts = await asyncio.gather(*[
                self._count_file_lines(file_path)
                for file_path in python_files
            ])
            
            stats['lines'] = sum(line_counts)
            
        except Exception as e:
            print(f"Error calculating stats: {str(e)}")
            
        return stats
    
    async def _find_python_files(self) -> List[str]:
        """Find Python files asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.fs_cache.executor,
            lambda: [
                str(path) for path in Path(self.root_path).rglob("*.py")
                if not any(pat in str(path) 
                          for pat in self.fs_cache.config.exclude_patterns)
            ]
        )
    
    async def _count_file_lines(self, file_path: str) -> int:
        """Count lines in file asynchronously"""
        try:
            content = await self.fs_cache.get_file_content(file_path)
            return content.count('\n') + 1
        except Exception:
            return 0








class LogAnalysisDashboard:
    """
    Comprehensive log analysis dashboard with advanced visualization and insights
    """
    
    def __init__(self, log_directory: str = 'logs'):
        """
        Initialize the log analysis dashboard
        
        Args:
            log_directory: Directory containing log files
        """
        self.log_directory = log_directory
        self.log_files = []
        self.log_data = []
        
        # Log level color mapping
        self.log_level_colors = {
            'DEBUG': '#87CEFA',     # Light Blue
            'INFO': '#90EE90',      # Light Green
            'WARNING': '#FFD700',   # Gold
            'ERROR': '#FF6347',     # Tomato Red
            'CRITICAL': '#8B0000'   # Dark Red
        }
        
        # Regular expression for parsing log entries
        self.log_pattern = re.compile(
            r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*'  # Timestamp
            r'\[(\w+)\]\s*'                               # Log Level
            r'(.*?)$'                                    # Message
        )
    
    def _load_log_files(self):
        """
        Load and parse log files
        """
        self.log_files = [
            f for f in os.listdir(self.log_directory) 
            if f.endswith('.log')
        ]
        
        self.log_data = []
        for log_file in self.log_files:
            full_path = os.path.join(self.log_directory, log_file)
            with open(full_path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = self.log_pattern.match(line.strip())
                    if match:
                        timestamp, level, message = match.groups()
                        self.log_data.append({
                            'timestamp': datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'),
                            'level': level,
                            'message': message,
                            'file': log_file
                        })
    
    def create_dash_app(self):
        """
        Create an interactive Dash application for log analysis
        """
        # Ensure log data is loaded
        self._load_log_files()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.log_data)
        
        # Initialize Dash app
        app = dash.Dash(__name__)
        
        # App layout
        app.layout = html.Div([
            html.H1('Log Analysis Dashboard', style={'textAlign': 'center'}),
            
            # Filters
            html.Div([
                html.Div([
                    html.Label('Select Log Level:'),
                    dcc.Dropdown(
                        id='level-dropdown',
                        options=[{'label': level, 'value': level} for level in df['level'].unique()],
                        multi=True,
                        value=df['level'].unique()
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label('Select Log Files:'),
                    dcc.Dropdown(
                        id='file-dropdown',
                        options=[{'label': file, 'value': file} for file in df['file'].unique()],
                        multi=True,
                        value=df['file'].unique()
                    )
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ]),
            
            # Visualizations
            html.Div([
                dcc.Graph(id='log-level-pie'),
                dcc.Graph(id='log-timeline'),
                dcc.Graph(id='log-file-distribution')
            ])
        ])
        
        # Callbacks for dynamic updates
        @app.callback(
            [Output('log-level-pie', 'figure'),
             Output('log-timeline', 'figure'),
             Output('log-file-distribution', 'figure')],
            [Input('level-dropdown', 'value'),
             Input('file-dropdown', 'value')]
        )
        def update_graphs(selected_levels, selected_files):
            # Filter data
            filtered_df = df[
                (df['level'].isin(selected_levels)) & 
                (df['file'].isin(selected_files))
            ]
            
            # Log Level Pie Chart
            level_counts = filtered_df['level'].value_counts()
            level_pie = px.pie(
                values=level_counts.values, 
                names=level_counts.index, 
                title='Log Levels Distribution',
                color_discrete_map=self.log_level_colors
            )
            
            # Log Timeline
            timeline = px.line(
                filtered_df, 
                x='timestamp', 
                color='level', 
                title='Log Events Over Time',
                color_discrete_map=self.log_level_colors
            )
            
            # Log File Distribution
            file_dist = px.bar(
                filtered_df['file'].value_counts(), 
                title='Log Entries per File',
                labels={'index': 'Log File', 'value': 'Number of Entries'}
            )
            
            return level_pie, timeline, file_dist
        
        return app
    
    def generate_log_summary(self):
        """
        Generate a comprehensive log summary report
        
        Returns:
            Markdown-formatted log summary
        """
        # Ensure log data is loaded
        self._load_log_files()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.log_data)
        
        # Create summary report
        summary = ["# Log Analysis Summary\n"]
        
        # Overall Statistics
        summary.append("## Overall Statistics")
        summary.append(f"- Total Log Entries: {len(df)}")
        summary.append(f"- Log Files Analyzed: {len(self.log_files)}")
        
        # Log Level Distribution
        summary.append("\n## Log Level Distribution")
        level_counts = df['level'].value_counts()
        for level, count in level_counts.items():
            summary.append(f"- {level}: {count} ({count/len(df)*100:.2f}%)")
        
        # Time-based Analysis
        summary.append("\n## Temporal Analysis")
        df['date'] = df['timestamp'].dt.date
        daily_counts = df.groupby(['date', 'level']).size().unstack(fill_value=0)
        summary.append("### Daily Log Level Breakdown")
        for date, row in daily_counts.iterrows():
            summary.append(f"- {date}: {dict(row)}")
        
        # File-specific Analysis
        summary.append("\n## Log File Analysis")
        file_level_counts = df.groupby(['file', 'level']).size().unstack(fill_value=0)
        for log_file, counts in file_level_counts.iterrows():
            summary.append(f"### {log_file}")
            for level, count in counts.items():
                summary.append(f"- {level}: {count}")
        
        return "\n".join(summary)










# IDK HOW I FEEL ABOUT THIS ONE, IS THIS EVEN USEFUL OR JUST CLUTTERING THIS OVER THE EDGE?
# like is this really needed ? Creates markdown-formatted imports report# 
# also is this like importing files or what? why would i want that 


class FileSummaryUtility:
    """
    Comprehensive utility for file analysis and summarization
    """
    
    @staticmethod
    def get_file_summary(
        file_path: str, 
        header_lines: int = 25, 
        footer_lines: int = 25, 
        max_file_size: int = 100_000
    ) -> str:
        """
        Create a smart summary of large files
        
        Args:
            file_path: Path to the file
            header_lines: Number of lines to show from the beginning
            footer_lines: Number of lines to show from the end
            max_file_size: Maximum file size to process
        
        Returns:
            Formatted file summary
        """
        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > max_file_size:
                return f"File too large: {file_size} bytes (max {max_file_size} bytes)"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Prepare summary
            summary = [
                f"=== File Summary: {file_path} ===\n",
                f"Total lines: {len(lines)}\n",
                "\n=== Header ===\n"
            ]
            
            # Add header lines
            summary.extend(lines[:header_lines])
            
            # Add truncation marker if needed
            if len(lines) > header_lines + footer_lines:
                summary.append("\n... (content truncated) ...\n")
            
            # Add footer lines
            summary.append("\n=== Footer ===\n")
            summary.extend(lines[-footer_lines:])
            
            return ''.join(summary)
        
        except Exception as e:
            return f"Error creating summary for {file_path}: {str(e)}"
    
    @staticmethod
    def extract_imports(file_path: str) -> Dict[str, List[str]]:
        """
        Extract and categorize imports from a Python file
        
        Args:
            file_path: Path to the Python file
        
        Returns:
            Dictionary of import types and their respective imports
        """
        imports = {
            'standard': [],      # import x
            'from_imports': [],  # from x import y
            'relative_imports': []  # from . import x or from ..module import y
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                # Standard imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports['standard'].append(alias.name)
                
                # From ... import statements
                elif isinstance(node, ast.ImportFrom):
                    # Determine import type
                    if node.level > 0:
                        # Relative import
                        import_source = '.' * node.level + (node.module or '')
                        for alias in node.names:
                            imports['relative_imports'].append(
                                f"{import_source}.{alias.name}" if node.module 
                                else f"{import_source}{alias.name}"
                            )
                    else:
                        # Standard from import
                        import_source = node.module or ''
                        for alias in node.names:
                            imports['from_imports'].append(
                                f"from {import_source} import {alias.name}"
                            )
            
            return imports
        
        except SyntaxError:
            return {"error": [f"Syntax error in {file_path}"]}
        except Exception as e:
            return {"error": [f"Error processing {file_path}: {str(e)}"]}
    
    @staticmethod
    def generate_imports_report(directory: str) -> str:
        """
        Generate a comprehensive report of imports across the project
        
        Args:
            directory: Root directory to scan for Python files
        
        Returns:
            Markdown-formatted imports report
        """
        imports_report = ["# Project Imports Report\n"]
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    
                    # Skip test and __init__ files
                    if any(x in file_path for x in ['test_', '__init__']):
                        continue
                    
                    file_imports = FileSummaryUtility.extract_imports(file_path)
                    
                    # Only add to report if imports exist
                    if any(file_imports.values()):
                        imports_report.append(f"## {file_path}\n")
                        
                        # Standard imports
                        if file_imports['standard']:
                            imports_report.append("### Standard Imports\n")
                            for imp in file_imports['standard']:
                                imports_report.append(f"- `import {imp}`\n")
                        
                        # From imports
                        if file_imports['from_imports']:
                            imports_report.append("### From Imports\n")
                            for imp in file_imports['from_imports']:
                                imports_report.append(f"- `{imp}`\n")
                        
                        # Relative imports
                        if file_imports['relative_imports']:
                            imports_report.append("### Relative Imports\n")
                            for imp in file_imports['relative_imports']:
                                imports_report.append(f"- `{imp}`\n")
                        
                        imports_report.append("\n")
        
        return "\n".join(imports_report)
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Union[str, int, List[str], List[int]]]:
        """
        Get comprehensive file information
        
        Args:
            file_path: Path to the file
        
        Returns:
            Dictionary with file information
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return {
                    'path': file_path,
                    'line_count': len(lines),
                    'content': lines,
                    'line_numbers': list(range(1, len(lines) + 1))
                }
        except Exception as e:
            print(f"\033[91m❌ Error reading file {file_path}: {str(e)}\033[0m")
            return {}
    
    @staticmethod
    def get_line_numbers(
        file_path: str, 
        search_term: Optional[str] = None
    ) -> List[int]:
        """
        Get line numbers for a file, optionally filtering by search term
        
        Args:
            file_path: Path to the file
            search_term: Optional term to search for
        
        Returns:
            List of line numbers
        """
        try:
            file_info = FileSummaryUtility.get_file_info(file_path)
            
            if not file_info:
                return []
            
            if search_term:
                return [
                    i + 1 for i, line in enumerate(file_info['content'])
                    if search_term in line
                ]
            
            return file_info['line_numbers']
        
        except Exception as e:
            print(f"\033[91mError getting line numbers: {e}\033[0m")
            return []
        








class EnhancedTreeVisualizer:
    """Enhanced tree visualization with detailed statistics"""
    
    ICONS = {
        'python': '🐍',
        'docker': '🐳',
        'config': '⚙️',
        'test': '🧪',
        'log': '📝',
        'yaml': '📄',
        'json': '📋',
        'md': '📚',
        'env': '🔑',
        'directory': '📁',
        'file': '📄'
    }
    
    FILE_COLORS = {
        '.py': '\033[96m',    # Cyan for Python
        '.yml': '\033[93m',   # Yellow for YAML
        '.json': '\033[95m',  # Magenta for JSON
        '.md': '\033[94m',    # Blue for Markdown
        '.env': '\033[92m',   # Green for env files
        '.log': '\033[91m',   # Red for logs
        'directory': '\033[94m',  # Blue for directories
        'default': '\033[37m'  # White for others
    }
    
    async def display_tree(self, path: str, stats_collector: 'ProjectStatsCollector'):
        """Enhanced tree display with statistics"""
        print("\n\033[96m📊 Project Structure Analysis\033[0m")
        print("=" * 50)
        
        # Display tree with enhanced visuals
        await self._display_tree_node(Path(path), "", True, stats_collector)
        
        # Show detailed statistics
        await self._display_detailed_stats(stats_collector)
    
    async def _display_tree_node(self, path: Path, prefix: str, is_last: bool, 
                               stats_collector: 'ProjectStatsCollector', level: int = 0):
        """Display individual tree node with enhanced visuals"""
        if level > 20:  # Prevent infinite recursion
            print(f"{prefix}...")
            return
            
        # Get node info
        node_name = path.name
        is_dir = path.is_dir()
        
        # Determine icon and color
        icon = self._get_icon(path)
        color = self._get_color(path)
        
        # Create connection lines
        connector = "└── " if is_last else "├── "
        
        # Display node with size info if it's a file
        if not is_dir:
            size_info = await self._get_file_size_info(path)
            print(f"{prefix}{connector}{color}{icon} {node_name} {size_info}\033[0m")
            await stats_collector.process_file(path)
        else:
            print(f"{prefix}{connector}{color}{icon} {node_name}/\033[0m")
            
            # Process directory contents
            try:
                contents = sorted(
                    [p for p in path.iterdir() 
                     if not any(pat in str(p) for pat in stats_collector.exclude_patterns)]
                )
                
                # Prepare next level prefix
                new_prefix = prefix + ("    " if is_last else "│   ")
                
                # Process contents
                for i, item in enumerate(contents):
                    await self._display_tree_node(
                        item,
                        new_prefix,
                        i == len(contents) - 1,
                        stats_collector,
                        level + 1
                    )
                    
            except Exception as e:
                print(f"{prefix}    \033[91mError reading directory: {str(e)}\033[0m")
    
    def _get_icon(self, path: Path) -> str:
        """Get appropriate icon for file/directory"""
        if path.is_dir():
            return self.ICONS['directory']
        
        ext = path.suffix.lower()
        if ext == '.py':
            return self.ICONS['python']
        elif ext in ['.yml', '.yaml']:
            return self.ICONS['yaml']
        elif ext == '.json':
            return self.ICONS['json']
        elif ext == '.md':
            return self.ICONS['md']
        elif ext == '.env':
            return self.ICONS['env']
        elif ext == '.log':
            return self.ICONS['log']
        
        return self.ICONS['file']
    
    def _get_color(self, path: Path) -> str:
        """Get appropriate color for file/directory"""
        if path.is_dir():
            return self.FILE_COLORS['directory']
            
        ext = path.suffix.lower()
        return self.FILE_COLORS.get(ext, self.FILE_COLORS['default'])
    
    async def _get_file_size_info(self, path: Path) -> str:
        """Get formatted file size information"""
        try:
            size = path.stat().st_size
            if size < 1024:
                return f"({size} B)"
            elif size < 1024 * 1024:
                return f"({size/1024:.1f} KB)"
            else:
                return f"({size/(1024*1024):.1f} MB)"
        except Exception:
            return ""






class ProjectStatsCollector:
    """Enhanced project statistics collector"""
    
    def __init__(self, exclude_patterns: List[str]):
        self.exclude_patterns = exclude_patterns
        self.stats = {
            'file_counts': defaultdict(int),
            'total_lines': 0,
            'code_lines': 0,
            'comment_lines': 0,
            'blank_lines': 0,
            'total_size': 0,
            'last_modified': datetime.min,
            'extensions': defaultdict(int),
            'authors': defaultdict(int)
        }
        
        self._comment_patterns = {
            '.py': r'^\s*#',
            '.yml': r'^\s*#',
            '.env': r'^\s*#',
            '.md': r'^\s*>',
        }
    
    async def process_file(self, path: Path):
        """Process individual file statistics"""
        try:
            # Basic file info
            ext = path.suffix.lower()
            stat = path.stat()
            
            # Update stats
            self.stats['file_counts'][ext] += 1
            self.stats['total_size'] += stat.st_size
            self.stats['extensions'][ext] += 1
            self.stats['last_modified'] = max(
                self.stats['last_modified'],
                datetime.fromtimestamp(stat.st_mtime)
            )
            
            # Detailed line analysis for text files
            if ext in ['.py', '.yml', '.yaml', '.md', '.env', '.json']:
                await self._analyze_file_content(path, ext)
                
        except Exception as e:
            print(f"\033[91mError processing stats for {path}: {str(e)}\033[0m")
    

    def _analyze_file_content(self, path: Path, ext: str):
            """Analyze file content for detailed statistics"""
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.splitlines()
                    
                    self.stats['total_lines'] += len(lines)
                    
                    # Analyze lines
                    comment_pattern = self._comment_patterns.get(ext)
                    if comment_pattern:
                        for line in lines:
                            stripped = line.strip()
                            if not stripped:
                                self.stats['blank_lines'] += 1
                            elif re.match(comment_pattern, line):
                                self.stats['comment_lines'] += 1
                            else:
                                self.stats['code_lines'] += 1
                                
            except Exception:
                pass
    
    async def get_formatted_stats(self) -> str:
        """Get formatted statistics output"""
        stats = []
        stats.append("\n\033[96m📊 Project Statistics\033[0m")
        stats.append("=" * 50)
        
        # File counts by type
        stats.append("\n\033[93mFile Distribution:\033[0m")
        for ext, count in sorted(self.stats['file_counts'].items()):
            stats.append(f"  • {ext}: {count} files")
        
        # Line counts
        stats.append("\n\033[93mCode Analysis:\033[0m")
        stats.append(f"  • Total Lines: {self.stats['total_lines']:,}")
        stats.append(f"  • Code Lines: {self.stats['code_lines']:,}")
        stats.append(f"  • Comment Lines: {self.stats['comment_lines']:,}")
        stats.append(f"  • Blank Lines: {self.stats['blank_lines']:,}")
        
        # Size information
        stats.append("\n\033[93mSize Information:\033[0m")
        stats.append(f"  • Total Size: {self._format_size(self.stats['total_size'])}")
        
        # Last modified
        stats.append("\n\033[93mTimestamp Information:\033[0m")
        stats.append(f"  • Last Modified: {self.stats['last_modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(stats)
    
    def _format_size(self, size: int) -> str:
        """Format size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"












class EnhancedLogAnalyzer:
    """Enhanced log analyzer with async capabilities and caching"""
    
    def __init__(self, log_dir: str = "logs", max_workers: int = 4):
        self.log_dir = Path(log_dir)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._log_cache: Dict[str, timedelta] = {}  # Cache for log content
        self._file_mtime_cache: Dict[str, float] = {}  # Cache for file modification times
        
        # Preserve original log categories
        self.LOG_CATEGORIES = {
            "development": [
                "backtester_development.log",
                "decorator_test_development.log",
                "level_test_development.log",
                "test_logger_development_development.log"
            ],
            "testing": [
                "database_service_testing.log",
                "decorator_test_testing.log",
                "market_data_repository_testing.log",
                "src.core.compression_manager_testing.log",
                "src.data.sources.yfinance_source_testing.log",
                "test_logger_testing_testing.log"
            ],
            "production": [
                "decorator_test_production.log",
                "test_logger_production_production.log"
            ],
            "database": [
                "sql_queries.log"
            ]
        }
        
        # Compile regular expressions once
        self.timestamp_pattern = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
        self.error_pattern = re.compile(r'ERROR|CRITICAL|EXCEPTION', re.IGNORECASE)
    
    @lru_cache(maxsize=100)
    def _should_reload_file(self, file_path: str) -> bool:
        """Check if file needs to be reloaded based on modification time"""
        try:
            current_mtime = Path(file_path).stat().st_mtime
            last_mtime = self._file_mtime_cache.get(file_path, 0)
            if current_mtime > last_mtime:
                self._file_mtime_cache[file_path] = current_mtime
                return True
            return False
        except Exception:
            return True
    
    async def get_recent_logs(self, time_window: str = "last") -> str:
        """Async version of get_recent_logs"""
        try:
            if time_window == "last":
                return await self._get_last_entries()
            
            since_time = await self._calculate_since_time(time_window)
            if since_time:
                return await self._get_logs_since(since_time)
            
            return "Invalid time_window specified"
            
        except Exception as e:
            return f"Error getting recent logs: {str(e)}"
    
    async def _get_last_entries(self) -> str:
        """Get last entry from each log file asynchronously"""
        log_files = list(self.log_dir.glob("*.log"))
        tasks = [self._get_last_entry(log_file) for log_file in log_files]
        entries = await asyncio.gather(*tasks)
        return "\n".join(filter(None, entries))
    
    async def _get_last_entry(self, log_file: Path) -> Optional[str]:
        """Get last entry from a single log file"""
        try:
            if not self._should_reload_file(str(log_file)):
                return self._log_cache.get(str(log_file))
                
            content = await self._read_file_async(log_file)
            lines = content.splitlines()
            if lines:
                result = f"=== {log_file.name} ===\n{lines[-1].strip()}"
                self._log_cache[str(log_file)] = result
                return result
            return None
        except Exception as e:
            return f"Error reading {log_file}: {str(e)}"
    
    async def _calculate_since_time(self, time_window: str) -> Optional[datetime]:
        """Calculate the since_time based on time_window"""
        now = datetime.now()
        
        if time_window == "today":
            return now.replace(hour=0, minute=0, second=0)
        elif time_window == "hour":
            return now - timedelta(hours=1)
        elif time_window == "day":
            return now - timedelta(days=1)
        elif time_window.endswith("min"):
            try:
                minutes = int(time_window[:-3])
                return now - timedelta(minutes=minutes)
            except ValueError:
                return None
        return None
    
    async def _read_file_async(self, file_path: Path) -> str:
        """Read file asynchronously using thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._read_file, file_path)
    
    def _read_file(self, file_path: Path) -> str:
        """Synchronous file reading helper"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    async def _get_logs_since(self, since_time: datetime) -> str:
        """Enhanced async version of _get_logs_since"""
        log_files = list(self.log_dir.glob("*.log"))
        tasks = [self._process_log_file(log_file, since_time) for log_file in log_files]
        entries = await asyncio.gather(*tasks)
        return "\n".join(filter(None, entries))
    
    async def _process_log_file(self, log_file: Path, since_time: datetime) -> Optional[str]:
        """Process a single log file asynchronously"""
        try:
            content = await self._read_file_async(log_file)
            recent_entries = await self._filter_recent_logs_async(content, since_time)
            
            if recent_entries:
                return f"=== {log_file.name} ===\n" + "\n".join(recent_entries)
            return None
        except Exception as e:
            return f"Error processing {log_file}: {str(e)}"
    
    async def _filter_recent_logs_async(self, content: str, since_time: datetime) -> List[str]:
        """Async version of _filter_recent_logs"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._filter_recent_logs,
            content,
            since_time
        )
    
    def _filter_recent_logs(self, content: str, since_time: datetime) -> List[str]:
        """Enhanced log filtering with better performance"""
        recent_entries = []
        for line in content.splitlines():
            match = self.timestamp_pattern.search(line)
            if match:
                try:
                    timestamp = datetime.strptime(match.group(), '%Y-%m-%d %H:%M:%S')
                    if timestamp >= since_time:
                        recent_entries.append(line.strip())
                except ValueError:
                    continue
        return recent_entries
    
    async def get_errors(self, hours: int = 24) -> str:
        """Enhanced async version of get_errors"""
        recent_time = datetime.now() - timedelta(hours=hours)
        log_files = list(self.log_dir.glob("*.log"))
        tasks = [self._get_file_errors(log_file, recent_time) for log_file in log_files]
        errors = await asyncio.gather(*tasks)
        return "\n".join(filter(None, errors))
    
    async def _get_file_errors(self, log_file: Path, since_time: datetime) -> Optional[str]:
        """Get errors from a single file asynchronously"""
        try:
            content = await self._read_file_async(log_file)
            errors = [
                f"{log_file.name}: {line.strip()}"
                for line in content.splitlines()
                if self.error_pattern.search(line)
            ]
            return "\n".join(errors) if errors else None
        except Exception as e:
            return f"Error processing {log_file}: {str(e)}"
        





class EnhancedContextLogger:
    def __init__(self, log_file='context_creation.log'):
        self.logger = logging.getLogger('ContextCreation')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))
        
        # Console handler with color
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_context_creation(self, 
                             context_type: str, 
                             files_processed: List[str], 
                             total_size: int):
        """Log detailed context creation details"""
        self.logger.info(
            f"Context Creation: Type={context_type}, "
            f"Files={len(files_processed)}, "
            f"Total Size={total_size} bytes"
        )
        
        # Optional: Log individual file details
        for file in files_processed:
            self.logger.debug(f"Processed File: {file}")









class EnhancedMenuSystem:
    """Enhanced menu system with async support and error handling"""
    
    def __init__(self, context_creator: 'EnhancedContextCreator'):
        self.context_creator = context_creator
        self.error_handler = context_creator.error_handler
        self.arch_guide = context_creator.arch_guide
        self.project_analysis = ProjectAnalysis(context_creator)
        self._current_section = None

    async def show_main_menu(self):
        """Display enhanced main menu with animations"""
        while True:
            try:
                await self._display_header()
                await self._display_stats()
                await self._display_menu_options()
                
                choice = await self._get_user_input("\n\033[92mEnter your choice (1-7):\033[0m ")
                
                if choice == "7":
                    print("\033[96m👋 Thank you for using AlgoBot Explorer!\033[0m")
                    break
                    
                await self._handle_menu_choice(choice)
                
            except KeyboardInterrupt:
                print("\n\033[93m⚠️ Operation cancelled by user\033[0m")
                continue
            except Exception as e:
                if self.error_handler:
                    await self.error_handler.handle_error(e, component="MenuSystem")
                print(f"\033[91m❌ Menu error: {str(e)}\033[0m")
                await self._get_user_input("Press Enter to continue...")

    async def _display_header(self):
        """Display menu header with ASCII art"""
        self.context_creator.clear_screen()
        print(f"\033[96m{self.context_creator.ascii_logo}\033[0m")

    async def _display_stats(self):
        """Display current project statistics"""
        stats = await self.context_creator._calculate_project_stats()
        print(f"\n\033[93m📊 Project Stats:\033[0m")
        print(f"   • Python Files: {stats['files']}")
        print(f"   • Total Lines: {stats['lines']}")

    async def _display_menu_options(self):
        """Display main menu options"""
        print("\n\033[96m=== Context Creator Menu ===\033[0m")
        print("1. 📄 Create Basic Context (Core Files)")
        print("2. 📦 Create Full Context (Including Logs)")
        print("3. 🔍 Create Custom Context")
        print("4. 📂 Browse Project Structure")
        print("5. 📊 View Architecture Guide")
        print("6. 💡 Show Project Analysis")
        print("7. ❌ Exit")

    async def _handle_menu_choice(self, choice: str):
        """Handle menu selections with error handling"""
        try:
            if choice == "1":
                context = await self.context_creator.create_context(["always"])
                await self.context_creator._save_context(context, "basic_context.md")
                print("\033[92m✅ Basic context created successfully!\033[0m")
                
            elif choice == "2":
                context = await self.context_creator.create_context(
                    ["always", "config", "docker"], 
                    include_logs=True
                )
                await self.context_creator._save_context(context, "full_context.md")
                print("\033[92m✅ Full context created successfully!\033[0m")
                
            elif choice == "3":
                await self._handle_custom_context()
                
            elif choice == "4":
                await self.show_file_browser()
                
            elif choice == "5":
                # View Architecture Guide (new method)
                await self.show_architecture_guide()
                
            elif choice == "6":
                await self.project_analysis.show_analysis_menu()
                
        except Exception as e:
            if self.error_handler:
                await self.error_handler.handle_error(e, component="MenuSystem")
            print(f"\033[91m❌ Error: {str(e)}\033[0m")
            await self._get_user_input("Press Enter to continue...")



    async def show_file_browser(self):
            """Enhanced file browser with async support"""
            current_path = "."
            while True:
                try:
                    self.context_creator.clear_screen()
                    print(f"\n\033[96m📂 File Browser: {current_path}\033[0m")
                    print("=" * 50)
                    
                    items = sorted([
                        item for item in os.listdir(current_path)
                        if not any(pat in item for pat in self.context_creator.exclude_patterns)
                    ])
                    
                    # Display items with numbers
                    for i, item in enumerate(items, 1):
                        full_path = os.path.join(current_path, item)
                        if os.path.isdir(full_path):
                            print(f"{i:2d}. \033[94m📂 {item}\033[0m")
                        else:
                            print(f"{i:2d}. \033[37m📄 {item}\033[0m")
                    
                    choice = await self._get_user_input("\n\033[92mEnter number, 'u' for up, or 'q' to quit:\033[0m ")
                    
                    if choice.lower() == 'q':
                        break
                    elif choice.lower() == 'u':
                        current_path = str(Path(current_path).parent)
                    elif choice.isdigit() and 1 <= int(choice) <= len(items):
                        selected = items[int(choice) - 1]
                        full_path = os.path.join(current_path, selected)
                        if os.path.isdir(full_path):
                            current_path = full_path
                        else:
                            await self.context_creator._display_file_preview(full_path)
                            await self._get_user_input("\nPress Enter to continue...")
                    
                except Exception as e:
                    if self.error_handler:
                        await self.error_handler.handle_error(e, "MenuSystem")
                    print(f"\033[91m❌ Error browsing files: {str(e)}\033[0m")
                    await self._get_user_input("Press Enter to continue...")

    async def show_architecture_guide(self):
        """Display interactive architecture guide from menu system"""
        try:
            while True:
                print("\n\033[96m=== Architecture Guide ===\033[0m")
                print("1. 🔄 System Flows")
                print("2. 📦 Component Details")
                print("3. 📊 Dependencies")
                print("4. 📖 Context Master Scheme")
                print("5. ↩️  Back")
                
                choice = await self._get_user_input("\n\033[92mEnter choice (1-5):\033[0m ")
                
                try:
                    if choice == "1":
                        # Show system flows (can use existing method from arch_guide)
                        await self.arch_guide._show_system_flows()
                    elif choice == "2":
                        # Show component details
                        await self.arch_guide._show_component_details()
                    elif choice == "3":
                        # Show dependencies
                        await self.arch_guide._show_dependencies()
                    elif choice == "4":
                        # Read and display context master scheme
                        try:
                            with open('context-master-scheme.md', 'r') as f:
                                print(f.read())
                            await self._get_user_input("\nPress Enter to continue...")
                        except FileNotFoundError:
                            print("\033[91m❌ Context master scheme not found.\033[0m")
                            await self._get_user_input("\nPress Enter to continue...")
                    elif choice == "5":
                        break
                except Exception as e:
                    if self.error_handler:
                        await self.error_handler.handle_error(e, "ArchitectureGuide")
                    print(f"\033[91m❌ Error in architecture guide: {str(e)}\033[0m")
                    await self._get_user_input("Press Enter to continue...")
                    
        except Exception as e:
            if self.error_handler:
                await self.error_handler.handle_error(e, "MenuSystem")
            print(f"\033[91m❌ Menu error: {str(e)}\033[0m")

    async def interactive_file_selection(self):
        """Guided, interactive file selection process"""                  # DOES THIS BELONG HERE????
        print("\n🔍 Interactive Context Creation")
        
        # Show file categories
        categories = list(self.config.path_mappings.keys())
        for i, category in enumerate(categories, 1):
            print(f"{i}. {category}")
        
        # Select categories
        category_choices = input("Select categories (comma-separated): ")
        selected_categories = [
            categories[int(c.strip())-1] 
            for c in category_choices.split(',')
        ]
        
        # Preview selected files
        selected_files = []
        for category in selected_categories:
            files = self.config.path_mappings[category]
            print(f"\n=== Files in {category} ===")
            for i, file in enumerate(files, 1):
                print(f"{i}. {file}")
            
            file_choices = input("Select files (comma-separated, or 'all'): ")
            if file_choices.lower() == 'all':
                selected_files.extend(files)
            else:
                selected_files.extend([files[int(c.strip())-1] for c in file_choices.split(',')])
        
        return selected_files







class ProjectAnalysis:
    """Project analysis functionality"""
    
    def __init__(self, context_creator: 'EnhancedContextCreator'):
        self.context_creator = context_creator
        self.error_handler = context_creator.error_handler

    async def show_analysis_menu(self):
        """Show project analysis menu"""
        while True:
            self.context_creator.clear_screen()
            print("\n\033[96m=== Project Analysis ===\033[0m")
            print("1. 📊 Code Statistics")
            print("2. 📁 File Distribution")
            print("3. 📈 Component Coverage")
            print("4. 🔍 Test Analysis")
            print("5. ↩️  Back to Main Menu")
            
            choice = await self._get_user_input("\n\033[92mEnter your choice (1-5):\033[0m ")
            
            if choice == "1":
                await self._show_code_statistics()
            elif choice == "2":
                await self._show_file_distribution()
            elif choice == "3":
                await self._show_component_coverage()
            elif choice == "4":
                await self._show_test_analysis()
            elif choice == "5":
                break

    async def _show_code_statistics(self):
        """Show detailed code statistics"""
        try:
            stats = await self.context_creator._calculate_project_stats()
            print("\n\033[93m=== Code Statistics ===\033[0m")
            print(f"Total Python Files: {stats['files']}")
            print(f"Total Lines of Code: {stats['lines']}")
            print(f"Average Lines per File: {stats['lines'] // stats['files'] if stats['files'] else 0}")
            
            # Add more detailed stats
            if 'comment_lines' in stats:
                print(f"Comment Lines: {stats['comment_lines']}")
            if 'blank_lines' in stats:
                print(f"Blank Lines: {stats['blank_lines']}")
            
            await self._get_user_input("\nPress Enter to continue...")
        except Exception as e:
            if self.error_handler:
                await self.error_handler.handle_error(e, "ProjectAnalysis")

    async def _show_file_distribution(self):
        """Show file type distribution"""
        try:
            distribution = await self._analyze_file_distribution()
            print("\n\033[93m=== File Distribution ===\033[0m")
            for ext, count in sorted(distribution.items()):
                print(f"{ext}: {count} files")
            await self._get_user_input("\nPress Enter to continue...")
        except Exception as e:
            if self.error_handler:
                await self.error_handler.handle_error(e, "ProjectAnalysis")

    async def _show_component_coverage(self):
        """Show component test coverage"""
        try:
            coverage = await self._analyze_component_coverage()
            print("\n\033[93m=== Component Coverage ===\033[0m")
            for component, stats in coverage.items():
                print(f"\n{component}:")
                print(f"  Files: {stats['files']}")
                print(f"  Tests: {stats['tests']}")
                print(f"  Coverage: {stats['coverage']}%")
            await self._get_user_input("\nPress Enter to continue...")
        except Exception as e:
            if self.error_handler:
                await self.error_handler.handle_error(e, "ProjectAnalysis")

    async def _show_test_analysis(self):
        """Show test statistics and analysis"""
        try:
            test_stats = await self._analyze_tests()
            print("\n\033[93m=== Test Analysis ===\033[0m")
            print(f"Total Test Files: {test_stats['total_files']}")
            print(f"Total Test Cases: {test_stats['total_cases']}")
            print("\nTest Distribution:")
            for category, count in test_stats['distribution'].items():
                print(f"  {category}: {count}")
            await self._get_user_input("\nPress Enter to continue...")
        except Exception as e:
            if self.error_handler:
                await self.error_handler.handle_error(e, "ProjectAnalysis")

    async def _get_user_input(self, prompt: str) -> str:
        """Get user input asynchronously"""
        return await self.context_creator._get_user_input(prompt)

    async def _analyze_file_distribution(self) -> dict:
        """Analyze file type distribution"""
        distribution = defaultdict(int)
        for root, _, files in os.walk(self.context_creator.base_dir):
            if not any(pat in root for pat in self.context_creator.exclude_patterns):
                for file in files:
                    ext = os.path.splitext(file)[1] or 'no extension'
                    distribution[ext] += 1
        return dict(distribution)

    async def _analyze_component_coverage(self) -> dict:
        """Analyze component test coverage"""
        # This is a placeholder implementation
        return {
            'core': {'files': 15, 'tests': 12, 'coverage': 80},
            'data': {'files': 10, 'tests': 8, 'coverage': 80},
            'trading': {'files': 20, 'tests': 15, 'coverage': 75}
        }

    async def _analyze_tests(self) -> dict:
        """Analyze test statistics"""
        # This is a placeholder implementation
        return {
            'total_files': 45,
            'total_cases': 250,
            'distribution': {
                'unit': 150,
                'integration': 75,
                'system': 25
            }
        }







class TerminalVisualization:
    """
    Terminal-based visualization utilities
    """
    
    def __init__(self):
        """Initialize console for rich output"""
        self.console = Console()
    
    def create_dependency_graph(self, dependency_map: dict):
        """
        Create a text-based dependency graph visualization
        
        Args:
            dependency_map: Dictionary of file dependencies
        """
        self.console.print("\n[bold blue]🕸️ Project Dependency Graph[/bold blue]\n")
        
        # Create a tree-like visualization
        tree = Tree("[bold green]Project Root[/bold green]")
        
        # Track processed files to avoid duplicates
        processed_files = set()
        
        for file, dependencies in dependency_map.items():
            # Shorten file path for readability
            short_file = os.path.basename(file)
            
            if file not in processed_files:
                # Add file node
                file_node = tree.add(f"[yellow]{short_file}[/yellow]")
                processed_files.add(file)
                
                # Add dependencies
                if dependencies:
                    dep_branch = file_node.add("[dim]Dependencies:[/dim]")
                    for dep in dependencies:
                        dep_short = os.path.basename(dep)
                        dep_branch.add(f"[cyan]→ {dep_short}[/cyan]")
        
        # Print the tree
        self.console.print(tree)
    
    def error_display(self, 
                      error: Exception, 
                      context: str = None, 
                      severity: str = "error"):
        """
        Create a visually appealing error display
        
        Args:
            error: Exception object
            context: Additional context for the error
            severity: Error severity level
        """
        # Severity-based icons and colors
        severity_map = {
            "critical": ("🚨", "bold red"),
            "error": ("❌", "red"),
            "warning": ("⚠️", "yellow"),
            "info": ("ℹ️", "blue")
        }
        
        icon, color = severity_map.get(severity.lower(), ("❌", "red"))
        
        # Create error panel
        error_panel = Panel(
            Text.assemble(
                (f"{icon} {type(error).__name__}: ", color),
                (str(error), "white")
            ),
            title=context or "Error Details",
            border_style=color,
            expand=False
        )
        
        self.console.print(error_panel)
        
        # Optional: Add traceback in verbose mode
        if severity in ["critical", "error"]:
            import traceback
            self.console.print(
                Panel(
                    Syntax(
                        traceback.format_exc(), 
                        "python", 
                        theme="monokai", 
                        line_numbers=True
                    ),
                    title="[dim]Traceback[/dim]",
                    border_style="dim",
                    expand=False
                )
            )
    
    def syntax_highlight_file(self, file_path: str, 
                               start_line: int = None, 
                               end_line: int = None):
        """
        Display syntax-highlighted file content
        
        Args:
            file_path: Path to the file
            start_line: Optional starting line number
            end_line: Optional ending line number
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Determine file type for syntax highlighting
            file_ext = os.path.splitext(file_path)[1].lstrip('.')
            
            # Create syntax-highlighted display
            syntax = Syntax(
                content, 
                file_ext, 
                theme="monokai", 
                line_numbers=True,
                start_line=start_line,
                end_line=end_line
            )
            
            self.console.print(
                Panel(
                    syntax, 
                    title=f"[bold blue]📄 {os.path.basename(file_path)}[/bold blue]",
                    border_style="blue"
                )
            )
        
        except Exception as e:
            self.error_display(
                e, 
                f"Error displaying {file_path}", 
                severity="warning"
            )
    
    def progress_bar(self, 
                     task_name: str, 
                     total_steps: int, 
                     update_callback=None):
        """
        Create an interactive progress bar
        
        Args:
            task_name: Name of the task
            total_steps: Total number of steps
            update_callback: Optional callback for each step
        """
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task(f"[green]{task_name}", total=total_steps)
            
            for i in range(total_steps):
                # Simulate work or call update callback
                if update_callback:
                    update_callback(i)
                
                progress.update(task, advance=1)
                # Optional: add a small delay to show progress
                import time
                time.sleep(0.1)









class EnhancedSystemArchitectureGuide:
    """Enhanced educational guide for system architecture and dependencies"""
    
    def __init__(self, config: dict, error_handler: Optional['ErrorHandler'] = None):
        self.error_handler = error_handler
        
        # Load from config
        arch_config = config.get('system_architecture', {})
        self.system_flows = arch_config.get('flows', {})
        self.component_relationships = arch_config.get('components', {})

    async def get_system_flow(self, flow_type: str) -> Optional[dict]:
        """Get system flow information with error handling"""
        try:
            return self.system_flows.get(flow_type)
        except Exception as e:
            if self.error_handler:
                await self.error_handler.handle_error(e, "SystemArchitectureGuide")
            return None

    async def get_component_guide(self, component: str) -> Optional[dict]:
        """Get architectural guidance for a component with error handling"""
        try:
            # Map component paths to their types
            component_map = {
                "src/core": "core",
                "src/data": "data",
                "src/trading": "trading",
                "src/ui": "ui"
            }
            
            component_type = next(
                (v for k, v in component_map.items() if k in component.lower()),
                None
            )
            
            if component_type:
                return self.component_relationships.get(component_type)
            return None
            
        except Exception as e:
            if self.error_handler:
                await self.error_handler.handle_error(e, "SystemArchitectureGuide")
            return None


    async def _show_system_flows(self):
        """Display system flows with formatting"""
        try:
            for name, flow in self.system_flows.items():
                print(f"\n\033[93m=== {name.title()} Flow ===\033[0m")
                print(f"\nPath: {flow['path']}")
                
                print("\nComponents:")
                for component in flow['components']:
                    print(f"  • {component}")
                
                print("\nKey Considerations:")
                for consideration in flow['key_considerations']:
                    print(f"  • {consideration}")
                
                print("\n" + "=" * 50)
            
            input("\nPress Enter to continue...")
            
        except Exception as e:
            if self.error_handler:
                await self.error_handler.handle_error(e, "SystemArchitectureGuide")

    async def _show_component_details(self):
        """Display component details with formatting"""
        try:
            while True:
                print("\n\033[96m=== Components ===\033[0m")
                for i, name in enumerate(self.component_relationships.keys(), 1):
                    print(f"{i}. {name.title()}")
                print(f"{len(self.component_relationships) + 1}. Back")
                
                choice = await self._get_user_input("\nSelect component (or 'back'): ")
                
                if choice.lower() == 'back' or choice == str(len(self.component_relationships) + 1):
                    break
                    
                try:
                    comp_name = list(self.component_relationships.keys())[int(choice) - 1]
                    comp_info = self.component_relationships[comp_name]
                    
                    print(f"\n\033[93m=== {comp_name.title()} Component ===\033[0m")
                    
                    for section, items in comp_info.items():
                        print(f"\n{section}:")
                        for item in items:
                            print(f"  • {item}")
                    
                    input("\nPress Enter to continue...")
                    
                except (IndexError, ValueError):
                    print("\033[91mInvalid choice\033[0m")
                    
        except Exception as e:
            if self.error_handler:
                await self.error_handler.handle_error(e, "SystemArchitectureGuide")

    async def _show_dependencies(self):
        """Display system dependencies with formatting"""
        try:
            print("\n\033[96m=== System Dependencies ===\033[0m")
            
            for comp_name, comp_info in self.component_relationships.items():
                print(f"\n\033[93m{comp_name.title()}\033[0m")
                
                if "dependencies" in comp_info:
                    print("\nDependencies:")
                    for dep in comp_info["dependencies"]:
                        print(f"  • {dep}")
                        
                if "dependent_components" in comp_info:
                    print("\nDependent Components:")
                    for dep in comp_info["dependent_components"]:
                        print(f"  • {dep}")
            
            input("\nPress Enter to continue...")
            
        except Exception as e:
            if self.error_handler:
                await self.error_handler.handle_error(e, "SystemArchitectureGuide")
        







class EnhancedContextCreator:
    """Enhanced context creator with improved performance and original functionality"""
    
    def __init__(self, base_dir: str = ".", config_path: str = "context_config.yaml"):
        
        # Error handling and logging
        self.logger = ErrorLogger()
        self.error_handler = IntegratedErrorHandler(self.logger)
        self.config = EnhancedConfigLoader().load_config(config_path)

        # new - wwhere to put?
        self.visualizer = TerminalVisualization()
        self.dependency_tracker = DependencyTracker(base_dir)
        self.file_summary_utility = FileSummaryUtility()    # are we keeping this one, for line number?
        self.visualizer.progress_bar("Creating Context", total_steps=10)
        #Terminal dashboard?


        # Configuration and caching
        self.config = EnhancedConfigLoader.load_config(config_path)
        self.fs_cache = FileSystemCache()
        self.events = ContextEventEmitter()
        
        # Original components
        self.base_dir = Path(base_dir)
        self.python_files = 0
        self.total_lines = 0
        self.log_analyzer = EnhancedLogAnalyzer()

        # Add architecture guide
        self.arch_guide = EnhancedSystemArchitectureGuide(self.error_handler)
        self.system_architecture = config.get('system_architecture', {})
        
        # ASCII art logo preserved
        self.ascii_logo = """
         █████╗ ██╗      ██████╗  ██████╗     ██████╗  ██████╗ ████████╗
        ██╔══██╗██║     ██╔════╝ ██╔═══██╗    ██╔══██╗██╔═══██╗╚══██╔══╝
        ███████║██║     ██║  ███╗██║   ██║    ██████╔╝██║   ██║   ██║   
        ██╔══██║██║     ██║   ██║██║   ██║    ██╔══██╗██║   ██║   ██║   
        ██║  ██║███████╗╚██████╔╝╚██████╔╝    ██████╔╝╚██████╔╝   ██║   
        ╚═╝  ╚═╝╚══════╝ ╚═════╝  ╚═════╝     ╚═════╝  ╚═════╝    ╚═╝   
        """
    



    
    async def create_context(self, 
                           context_type: str, 
                           include_logs: bool = False,
                           log_window: str = "last") -> str:
        """Create context with enhanced error handling"""
        try:
            return await self.error_handler.handle_operation(
                self._create_context_internal,
                context_type,
                include_logs,
                log_window,
                operation_name=f"create_context_{context_type}",
                dependency_report = self.dependency_tracker.generate_dependency_report(),
                 # Option to include log analysis summary
                log_summary = self.log_dashboard.generate_log_summary(),
                  # Option to add line numbers to files
                context_with_line_numbers = self.line_numberer.add_line_numbers(context),     #context_type????
                file_summaries = []
                for file_path in relevant_files:
                    file_summaries.append(
                        self.file_summary_utility.get_file_summary(file_path)
                    ),

                ########
                # Option to add imports report - JUST APPEND IT TO THE CONTEXT??? OR PUT IT IN CORE_FILES AND ADD ASS A DEV/DEBUG MODE???
                ########
                imports_report = self.file_summary_utility.generate_imports_report(self.base_dir)
                    )
        except Exception as e:
            self.logger.log_error(e, {
                'context_type': context_type,
                'include_logs': include_logs,
                'log_window': log_window
            })
            self.events.emit("context_error", {"error": str(e)})
            raise




    async def _create_context_internal(self, 
                                     context_type: str,
                                     include_logs: bool,
                                     log_window: str) -> str:
        """Internal context creation logic"""
        try:
            self.events.emit("context_start", {"type": context_type})
            
            # Get paths from config
            paths = self.config.path_mappings.get(context_type, [])
            if not paths:
                self.events.emit("warning", {
                    "message": f"No paths found for {context_type}, using minimal context"
                })
                paths = self.config.path_mappings.get("minimal", [])
            
            # Process files in parallel
            tasks = [
                self.process_file(path)
                for path in paths
                if not any(pat in path for pat in self.config.exclude_patterns)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Build context
            context_parts = []
            context_parts.extend(filter(None, results))
            
            # Add logs if requested
            if include_logs:
                logs = await self.log_analyzer.get_recent_logs(log_window)
                context_parts.append("\n=== Recent Logs ===\n")
                context_parts.append(logs)
            
            # Add architecture guide
            if context_type in self.config.component_relationships:
                context_parts.append("\n=== Architecture Guide ===\n")
                context_parts.append(self._get_architecture_guide(context_type))
            
            context = "\n".join(context_parts)
            
            self.events.emit("context_complete", {
                "type": context_type,
                "size": len(context)
            })
            
            return context
            
        except Exception as e:
            self.logger.log_error(e, {
                'operation': '_create_context_internal',
                'context_type': context_type,
                'include_logs': include_logs
            })
            raise ContextError(f"Error creating context: {str(e)}")
    



    async def process_file(self, file_path: str) -> Optional[str]:
        """Process a single file with error handling"""
        try:
            if not await FileValidator.validate_file_access(file_path):
                self.events.emit("warning", {
                    "message": f"Skipping inaccessible file: {file_path}"
                })
                return None

            content = await self.fs_cache.get_file_content(file_path)
            if not content:
                return None

            # Format the content with file path header
            return f"\n=== {file_path} ===\n{content}\n"

        except Exception as e:
            await self.error_handler.handle_error(e, f"process_file: {file_path}")
            return None


    async def _display_file_preview(self, file_path: str, lines: int = 10):
        """Display file preview with syntax highlighting"""
        try:
            if not await FileValidator.validate_file_access(file_path):
                print(f"\033[91m❌ Cannot access file: {file_path}\033[0m")
                return

            content = await self.fs_cache.get_file_content(file_path)
            if not content:
                return

            print(f"\n\033[96m=== File Preview: {file_path} ===\033[0m")
            print("=" * 50)

            # Split and limit lines
            preview_lines = content.splitlines()[:lines]
            
            # Basic syntax highlighting for Python files
            if file_path.endswith('.py'):
                import re
                for line in preview_lines:
                    # Keywords in yellow
                    line = re.sub(
                        r'\b(def|class|import|from|return|if|else|elif|try|except|finally)\b',
                        '\033[93m\\1\033[0m',
                        line
                    )
                    # Comments in green
                    line = re.sub(r'(#.*$)', '\033[92m\\1\033[0m', line)
                    # Strings in cyan
                    line = re.sub(r'([\'"].*?[\'"])', '\033[96m\\1\033[0m', line)
                    print(f"  {line}")
            else:
                for line in preview_lines:
                    print(f"  {line}")
            if len(preview_lines) < content.count('\n'):
                print("  ...")
            print("=" * 50)
        except Exception as e:
            print(f"\033[91m❌ Error displaying preview: {str(e)}\033[0m")


    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    

    async def show_loading_animation(self):
        """Async loading animation"""
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        for _ in range(10):
            for frame in frames:
                print(f"\r\033[96m{frame} Loading AlgoBot Explorer...\033[0m", end="")
                await asyncio.sleep(0.1)
        print("\n")
    

    async def _display_directory_tree(self):
        """Display directory structure asynchronously"""
        self.clear_screen()
        print(f"\033[96m{self.ascii_logo}\033[0m")
        print("\n\033[96m📁 Project Directory Structure\033[0m")
        print("=" * 50)
        
        # Process directory structure asynchronously
        structure = await self._build_directory_structure()
        await self._display_structure(structure)
        
        # Show project stats
        stats = await self._calculate_project_stats()
        print("\n\033[93mProject Statistics:\033[0m")
        print(f"Python Files: {stats['files']}")
        print(f"Total Lines: {stats['lines']}")
    

    async def _build_directory_structure(self) -> Dict[str, Any]:
        """Build directory structure asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.fs_cache.executor,
            self._scan_directory,
            self.base_dir
        )
    

    def _scan_directory(self, path: Path) -> Dict[str, Any]:
        """Scan directory synchronously"""
        structure = {"files": []}
        try:
            for item in path.iterdir():
                if any(pat in str(item) for pat in self.config.exclude_patterns):
                    continue
                    
                if item.is_file():
                    structure["files"].append(item.name)
                elif item.is_dir():
                    structure[item.name] = self._scan_directory(item)
                    
        except Exception as e:
            self.events.emit("scan_error", {
                "path": str(path),
                "error": str(e)
            })
            
        return structure
    
            
    async def _get_user_input(self, prompt: str) -> str:
        """Get user input asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, input, prompt)
   

    async def _handle_custom_context(self):
        """Enhanced async custom context handler"""
        print("\nAvailable categories:", ", ".join(self.config.core_files.keys()))
        categories = (await self._get_user_input("Enter categories (comma-separated): ")).split(',')
        include_logs = (await self._get_user_input("Include logs? (y/n): ")).lower() == 'y'
        
        try:
            context = await self.create_context(
                [cat.strip() for cat in categories],
                include_logs=include_logs
            )
            await self._save_context(context, "custom_context.md")
            print("\033[92m✅ Custom context created successfully!\033[0m")
        except Exception as e:
            self.events.emit("error", {
                "operation": "custom_context",
                "error": str(e)
            })
            print(f"\033[91m❌ Error creating custom context: {str(e)}\033[0m")


    async def _save_context(self, context: str, filename: str):
        """Enhanced async context saving"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.fs_cache.executor,
                self._write_file,
                filename,
                context
            )
            print(f"\nContext saved to {filename}")
        except Exception as e:
            self.events.emit("error", {
                "operation": "save_context",
                "filename": filename,
                "error": str(e)
            })
            raise


    def _write_file(self, filename: str, content: str):
        """Synchronous file writing helper"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)


    async def _get_user_input(self, prompt: str) -> str:
        """Async wrapper for user input"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, input, prompt)


    @staticmethod
    def _format_imports(imports: Dict[str, List[str]]) -> str:
        """Format imports for display"""
        return "\n".join(
            f"{file}:\n" + "\n".join(f"  {imp}" for imp in imps)
            for file, imps in imports.items()
        )


    async def _cleanup(self):
        """Enhanced async cleanup"""
        try:
            self.fs_cache.executor.shutdown(wait=True)
            # Add any additional cleanup here
        except Exception as e:
            self.events.emit("error", {
                "operation": "cleanup",
                "error": str(e)
            })
            print(f"\033[91m⚠️ Cleanup error: {str(e)}\033[0m")



    async def _show_selection_preview(self):
        """Show preview of selected components"""
        try:
            print("\n\033[96m=== Selection Preview ===\033[0m")
            
            # Initialize stats collector
            self.stats_collector = ProjectStatsCollector(self.config.exclude_patterns)
            
            # Get paths for selected components
            paths = self._get_paths_for_selection()
            
            # Show tree visualization for selected paths
            for path in paths:
                await self.tree_visualizer.display_tree(path, self.stats_collector)
            
            # Show statistics
            print(await self.stats_collector.get_formatted_stats())
            
        except Exception as e:
            print(f"\033[91mError showing preview: {str(e)}\033[0m")
    

    def _get_paths_for_selection(self) -> List[str]:
        """Get file paths for selected components"""
        paths = []
        for component in self.current_selection:
            component_paths = self.config.path_mappings.get(component, [])
            paths.extend(component_paths)
        return list(set(paths))  # Remove duplicates
    

    async def _confirm_selection(self) -> bool:
        """Confirm current selection"""
        confirm = await self._get_user_input(
            "\n\033[93mConfirm this selection? (y/n): \033[0m"
        )
        return confirm.lower() == 'y'
    

    @staticmethod
    async def _get_user_input(prompt: str) -> str:
        """Get user input asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, input, prompt)
    

    @staticmethod
    def clear_screen():
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')









##################################
# This is a list of soome context during developing that may be helpful   
###################################

#Terminal dashbaord?

#What else new but isnt in here?



# # Example usage:
# async def main():
#     creator = EnhancedContextCreator()
    
#     # Subscribe to events
#     def log_progress(event_type: str, data: dict):
#         print(f"Event: {event_type}, Data: {data}")
#     creator.events.subscribe(log_progress)
    
#     # Create context
#     context = await creator.create_context("core_infrastructure")
    
#     # Save context
#     with open("context.md", "w") as f:
#         f.write(context)


# # EXAMPLE USE AFTER ADDING CONTEXTMANAGER. NEED TO INTEGRATE STILL!!!
#     manager = ContextManager()
#     if await manager.initialize():
#         # Create and export context
#         context = await manager.create_context("core_infrastructure", include_logs=True)
#         if context:
#             await manager.export_context()
        
#         # Get operation summary
#         summary = await manager.get_operation_summary()
#         print("\nOperation Summary:")
#         print(json.dumps(summary, indent=2))


# # Example usage (FROM DEPENDENCY TRACKER)
# def main():
#     # Initialize the tracker
#     tracker = DependencyTracker("/path/to/your/project")
    
#     # Generate dependency graph visualization
#     tracker.visualize_dependency_graph()
    
#     # Print dependency report
#     print(tracker.generate_dependency_report())
    
#     # Optionally, get detailed dependency map
#     dependency_map = tracker.build_dependency_map()
#     print("\nDetailed Dependency Map:")
#     for file, deps in dependency_map.items():
#         print(f"{file}: {deps}")


# # Example usage - LOG ANALYSIS DASHBOARD???
# def main():
#     # Create dashboard instance
#     dashboard = LogAnalysisDashboard('/path/to/logs')
    
#     # Generate and print summary report
#     print(dashboard.generate_log_summary())
    
#     # Create and run Dash app
#     app = dashboard.create_dash_app()
#     app.run_server(debug=True)


# # Example usage - file line numberer
# def main():
#     # Line numbering example
#     file_path = "example.py"
    
#     # Add line numbers to entire file
#     with open(file_path, 'r') as f:
#         content = f.read()
#         numbered_content = FileLineNumberer.add_line_numbers(content)
#         print(numbered_content)
    
#     # Find lines containing specific terms
#     function_lines = FileLineNumberer.find_line_numbers(file_path, "def ")
#     print("\nFunction definition lines:", function_lines)
    
#     # Extract specific file sections
#     section = FileLineNumberer.extract_file_section(file_path, 10, 20)
#     print("\nFile Section (Lines 10-20):")
#     print(section)


# # Example usage   -  filesummaryutility, dont know if im a fan of this one, maybe be unnecessary (EXCEPT THE LINE COUNTING/NUMBERING!!!)
# def main():
#     # File summary example
#     print(FileSummaryUtility.get_file_summary('example.py'))
    
#     # Imports extraction example
#     imports = FileSummaryUtility.extract_imports('example.py')
#     print("\nImports:")
#     for category, imp_list in imports.items():
#         print(f"{category.upper()}:")
#         for imp in imp_list:
#             print(f"  - {imp}")
    
#     # Generate project-wide imports report
#     project_imports_report = FileSummaryUtility.generate_imports_report('/path/to/project')
#     with open('project_imports_report.md', 'w') as f:
#         f.write(project_imports_report)
    
#     # Line number examples
#     print("\nAll function definition lines:")
#     print(FileSummaryUtility.get_line_numbers('example.py', 'def '))
    
#     # Full project imports report
#     print("\nGenerated project imports report at project_imports_report.md")



# #### Example usage   -  t emrinal visualization....
# def main():
#     viz = TerminalVisualization()
    
#     # Dependency graph example
#     dependency_map = {
#         "/path/to/main.py": ["/path/to/config.py", "/path/to/utils.py"],
#         "/path/to/config.py": ["/path/to/settings.py"],
#         "/path/to/utils.py": []
#     }
#     viz.create_dependency_graph(dependency_map)
    
#     # Error display example
#     try:
#         # Intentional error to demonstrate
#         1 / 0
#     except Exception as e:
#         viz.error_display(e, "Division by Zero Error", "critical")
    
#     # Syntax highlighting example
#     viz.syntax_highlight_file("example.py")
    
#     # Progress bar example
#     def dummy_work(step):
#         print(f"Processing step {step}")
    
#     viz.progress_bar("Context Generation", 10, dummy_work)


# #OLDER VERSION REFERENCE (BEFORE THE MASS RESTRUCTURE/ENHANCEMENTS)
# #####
# # ALSO GIVE THEM THE CONTEXTCREATOR OLD CLASS AS WELL AS THAT HAD A LOT OF "MAIN" FUNCTIONALITY IN IT!!!
# ######

# def main():
#     """Main entry point with error handling and command line arguments"""
#     import argparse
        
#     parser = argparse.ArgumentParser(description="AlgoBot Context Creator")
#     parser.add_argument("--level", choices=list(ContextLevels.LEVELS.keys()),
#                     help="Context level to use")
#     parser.add_argument("--logs", action="store_true",
#                     help="Include logs in context")
#     parser.add_argument("--log-window", default="last",
#                     choices=["last", "hour", "today", "day"],
#                     help="Time window for logs")
#     parser.add_argument("--output", default="context.md",
#                     help="Output file name")
        
#     args = parser.parse_args()
        
#     try:
#         creator = ContextCreator()
            
#         if args.level:
#             # Use specified context level
#             level_config = ContextLevels.get_level(args.level)
#             context = creator.create_context(
#                 include_categories=level_config["categories"],
#                 include_logs=level_config.get("include_logs", False),
#                 log_window=level_config.get("log_window", "last")
#             )
#             creator._save_context(context, args.output)
#             print(f"\033[92m✅ Created {args.level} context in {args.output}\033[0m")
                
#         else:
#                 # Start interactive menu
#             creator.start()
                
#     except KeyboardInterrupt:
#         print("\n\033[93m⚠️ Operation cancelled by user\033[0m")
#     except Exception as e:
#         print(f"\033[91m❌ Error: {str(e)}\033[0m")
#         import traceback
#         traceback.print_exc()


# if __name__ == "__main__":
#     asyncio.run(main())