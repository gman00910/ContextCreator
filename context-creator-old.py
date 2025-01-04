import glob
import os
import re
import time
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class LogAnalyzer:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
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
        
    def get_recent_logs(self, minutes: int = 5) -> str:
        """Get logs from last N minutes"""
        recent_time = datetime.now() - timedelta(minutes=minutes)
        relevant_logs = []
        
        for log_file in self.log_dir.glob("*.log"):
            with open(log_file) as f:
                # Get last N lines that are recent enough
                recent_entries = self._filter_recent_logs(f, recent_time)
                if recent_entries:
                    relevant_logs.append(f"=== {log_file.name} ===")
                    relevant_logs.extend(recent_entries)
        
        return "\n".join(relevant_logs)

    def get_errors(self, hours: int = 24) -> str:
        """Get all errors from last N hours"""
        error_pattern = re.compile(r'ERROR|CRITICAL|EXCEPTION', re.IGNORECASE)
        recent_time = datetime.now() - timedelta(hours=hours)
        errors = []
        
        for log_file in self.log_dir.glob("*.log"):
            with open(log_file) as f:
                for line in f:
                    if error_pattern.search(line):
                        errors.append(f"{log_file.name}: {line.strip()}") # timestamp check 
                        
        return "\n".join(errors)

    @staticmethod
    def _filter_recent_logs(file_handle, since_time):
        # Implementation to parse log timestamps and filter
        pass


class DirectoryStructure:
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        
    def build_tree(self) -> Dict[str, Any]:
        """Dynamically builds directory structure"""
        tree = {
            "src": {"files": []},
            "tests": {"files": []},
            "config": {"files": []},
            "scripts": {"files": []},
            "logs": {"files": []},
            "docs": {"files": []},
            "docker": {"files": []},
        }
        
        def add_to_tree(path: Path, structure: Dict[str, Any]):
            relative = path.relative_to(self.root_path)
            parts = relative.parts
            
            current = structure
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {"files": []}
                current = current[part]
                
            if path.is_file():
                current["files"].append(parts[-1])
            elif path.is_dir() and parts:
                if parts[-1] not in current:
                    current[parts[-1]] = {"files": []}
        
        # Recursively scan directory
        for path in self.root_path.rglob('*'):
            if '__pycache__' not in str(path):
                add_to_tree(path, tree)
                
        return tree

    def display_structure(self):
        """Display structured tree view of components"""
        print("\n📁 ALGOBOT STRUCTURE")
        print("==================")
        
        tree = self.build_tree()
        
        def print_tree(structure: Dict[str, Any], prefix: str = ""):
            for key, value in sorted(structure.items()):
                if key != "files":
                    print(f"{prefix}├── {key}")
                    if "files" in value and value["files"]:
                        for file in sorted(value["files"]):
                            print(f"{prefix}│   ├── {file}")
                    print_tree(value, prefix + "│   ")
        
        print_tree(tree)
        print("\n")

class SystemArchitectureGuide:
    """Educational guide for system architecture and dependencies"""
    def __init__(self):
        self.architecture_patterns = {
            "Event-Driven Core": {
                "description": "Central event system architecture",
                "key_components": [
                    "src/core/event_bus.py",
                    "src/core/bot.py",
                    "src/trading_system.py"
                ],
                "flow": [
                    "Events published to event_bus",
                    "Components subscribe to relevant events",
                    "Event handlers process and respond"
                ],
                "best_practices": [
                    "Use typed events",
                    "Implement error boundaries",
                    "Maintain event documentation"
                ]
            },
            "Data Pipeline": {
                "description": "Market data flow and processing",
                "key_components": [
                    "src/data/sources/*",
                    "src/data/validation.py",
                    "src/data/repository.py"
                ],
                "flow": [
                    "Source → Validation → Processing → Storage",
                    "Real-time updates through event system",
                    "Cached access via repository"
                ],
                "best_practices": [
                    "Validate data at entry",
                    "Implement caching strategy",
                    "Monitor data quality"
                ]
            },
            "Trading Engine": {
                "description": "Core trading logic and execution",
                "key_components": [
                    "src/strategies/*",
                    "src/backtesting/*",
                    "src/engines/execution/*"
                ],
                "flow": [
                    "Strategy signals → Risk checks → Order execution",
                    "Position management and monitoring",
                    "Performance tracking and analysis"
                ],
                "best_practices": [
                    "Implement risk controls",
                    "Log all decisions",
                    "Monitor performance metrics"
                ]
            }
        }
        
        self.system_flows = {
            "Market Data": {
                "path": "Source → Validation → Storage → Analysis",
                "components": ["data_source", "validator", "repository", "analyzer"],
                "key_considerations": [
                    "Data quality",
                    "Processing speed",
                    "Storage efficiency"
                ]
            },
            "Trading": {
                "path": "Signal → Validation → Execution → Monitoring",
                "components": ["strategy", "risk_manager", "broker", "monitor"],
                "key_considerations": [
                    "Risk management",
                    "Execution speed",
                    "Position tracking"
                ]
            }
        }

    def get_component_guide(self, component: str) -> dict:
        """Get architectural guidance for a component"""
        if component in self.architecture_patterns:
            return self.architecture_patterns[component]
        return None

    def get_system_flow(self, flow_type: str) -> dict:
        """Get system flow information"""
        if flow_type in self.system_flows:
            return self.system_flows[flow_type]
        return None

    def generate_documentation(self) -> str:
        """Generate system architecture documentation"""
        doc = ["# System Architecture Guide\n"]
        
        # Add patterns
        doc.append("## Core Patterns\n")
        for pattern, details in self.architecture_patterns.items():
            doc.append(f"### {pattern}\n")
            doc.append(f"{details['description']}\n")
            doc.append("#### Key Components:\n")
            for comp in details['key_components']:
                doc.append(f"- {comp}\n")
            doc.append("\n#### Flow:\n")
            for flow_step in details['flow']:
                doc.append(f"- {flow_step}\n")
            doc.append("\n#### Best Practices:\n")
            for practice in details['best_practices']:
                doc.append(f"- {practice}\n")
            doc.append("\n")
        
        # Add system flows
        doc.append("## System Flows\n")
        for flow_name, flow_details in self.system_flows.items():
            doc.append(f"### {flow_name}\n")
            doc.append(f"Path: {flow_details['path']}\n")
            doc.append("\nComponents:\n")
            for comp in flow_details['components']:
                doc.append(f"- {comp}\n")
            doc.append("\nKey Considerations:\n")
            for consideration in flow_details['key_considerations']:
                doc.append(f"- {consideration}\n")
            doc.append("\n")
        
        return "\n".join(doc)



class ContextCreator:
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.python_files = 0
        self.total_lines = 0
        self.arch_guide = SystemArchitectureGuide()  # Architecture guide hehe # Add core files to each section
        self.ascii_logo = """
         █████╗ ██╗      ██████╗  ██████╗     ██████╗  ██████╗ ████████╗
        ██╔══██╗██║     ██╔════╝ ██╔═══██╗    ██╔══██╗██╔═══██╗╚══██╔══╝
        ███████║██║     ██║  ███╗██║   ██║    ██████╔╝██║   ██║   ██║   
        ██╔══██║██║     ██║   ██║██║   ██║    ██╔══██╗██║   ██║   ██║   
        ██║  ██║███████╗╚██████╔╝╚██████╔╝    ██████╔╝╚██████╔╝   ██║   
        ╚═╝  ╚═╝╚══════╝ ╚═════╝  ╚═════╝     ╚═════╝  ╚═════╝    ╚═╝   
        """
        self.CORE_FILES = {
            "always": [  # Always included
                "src/main.py",
                "src/trading_system.py",
                "manage.py",
                "run_dashboard.py",
                "setup.py",
                "README.md"
            ],
            "docker": [  # Docker-related
                "docker-compose.base.yml",
                "docker-compose.dev.yml", 
                "docker-compose.prod.yml",
                "docker/dockerfile",
                "docker/dockerfile.dev",
                "docker/entrypoint.sh",
                "docker/timescaledb-init.sql"
            ],
            "env": [  # Environment files
                ".env.example",   #why example for "version control"???   AND env.test too?
                ".env.development", # Environment templates (not actual .env files) -->   WHYYYY
                ".env.production",
                ".env.docker.development",
                ".env.docker",
                ".env.test,          
            ],
            "logs": [],                             # Placeholder for dynamic log files
            "plan": [ "context-master-scheme.md" ], # MY MASTER PLAN IN MD FORMAT (TRY TO KEEP THIS UP TO DATE)
        }

        self.log_analyzer = LogAnalyzer()  # Integrated but separate concern

        self.exclude_patterns = [
            '*.pyc',
            '*/__pycache__/*',
            '*.git*',
            '*/.pytest_cache*',
            '*/logs/*',
            '*/.vscode/*',
            '*.log',
            '*/venv/*',
            '*.tmp',
            '*.temp'
        ]

        # Single comprehensive path map with tests properly mapped
        self.path_map = {
            "1": [  # Core Infrastructure 🎯
                ".env*",  # Include all environment files
                "docker-compose*", # Include all docker environment files
                "config/**/*.py",
                "src/config/**/*.py",
                "src/core/**/*.py",
                "tests/core/*.py",  # All core tests
                "tests/integration/*.py",  # Integration tests relevant to core
                "tests/infrastructure/*.py"  # infrastructure tests
            ],
            "2": [  # Data Pipeline & Storage 🔄
                "config/**/*.py",
                "src/config/**/*.py",
                "src/data/sources/**/*.py",  # All data sources
                "src/data/**/*.py", # All data-related files
                "tests/core/**/*.py",  # Core system tests
                "tests/integration/*.py",  # System integration tests
                "tests/infrastructure/*.py",  # infrastructure tests
                "tests/utils/init_test_db.py"  # Test database setup
            ],
            "3": [  # Trading Engine 📈
                "config/**/*.py",
                "src/config/**/*.py",
                "src/strategies/**/*.py",
                "src/backtesting/**/*.py",
                "src/engines/execution/order_entry.py",
                "tests/trading/*",  # All trading tests
                "tests/integration/test_trading_flow.py"
            ],
            "4": [  # Analysis Engine 🔬
                ".env*",  # Include all environment files
                "docker-compose*", # Include all docker environment files
                "config/**/*.py",
                "src/config/**/*.py",
                "tests/core/**/*.py",  # Core system tests
                "src/engines/**/*.py",
                "src/data/sources/interfaces.py",
                "tests/utils/**/*.py",  # Core system tests
            ],
            "5": [  # UI & Dashboard 🎨
                "run_dashboard.py",  # Dashboard entry point
                "config/**/*.py",
                "src/config/**/*.py",
                "src/ui/**/*.py"
                "src/core/event_bus.py",  # UI needs event system
                "src/core/logging.py",    # UI needs logging
                "src/core/db_services.py",    # UI needs logging
                "src/data_repository.py",    # UI needs logging
                "tests/integration/test_ui_integration.py"  # Future integration tests
            ],
            "6": [  # System Configuration ⚙️
                ".env*",  # Include all environment files
                "docker-compose*", # Include all docker environment files
                "config/**/*",
                "src/config/**/*.py",
                "src/config/.env",
                "src/core/**/*.py",
                "src/core/logging.py",
                "tests/core/test_config*.py",
                "tests/integration/test_config*.py",
                "tests/infrastructure/test_env*.py",

                # "docker-compose.base.yml", #The main docker that the other build from
                # "docker-compose.dev.yml",
                # "docker-compose.prod.yml",
            ],
            "7": [  # Development Tools 🛠️
                ".env*",  # Include all environment files
                "docker-compose*", # Include all docker environment files
                "config/**/*.py",
                "scripts/*.py"
                "src/config/**/*.py",
                "src/core/debug.py",
                "src/core/integration.py",
                "src/core/logging.py",
                "src/core/debug.py",
                "tests/**/*.py",  # Include ALL tests for development tools
                "scripts/**/*.py", 
                "context-creator.py",
            ],
            "8": [  # Runtime Core 🚀
                ".env*",  # Include all environment files
                "docker-compose*", # Include all docker environment files
                "config/**/*.py",
                "src/config/**/*.py",
                "tests/core/**/*.py",
                "scripts/**/*.py",
            ],
            "9": [  # Test Structure 🧪
                ".env*",  # Include all environment files
                "docker-compose*", # Include all docker environment files
                "config/**/*.py",
                "src/config/**/*.py",
                "tests/core/**/*.py",           # Core system tests
                "tests/data/**/*.py",          # Data management tests
                "tests/infrastructure/**/*.py", # Infrastructure tests
                "tests/integration/**/*.py",    # Integration tests
                "tests/trading/**/*.py",       # Trading system tests
                "tests/utils/**/*.py"          # Test utilities
                "scripts/**/*.py"                 # SCRIPT utilities (utility scripts, deployment tools, etc.)
                "context-creator.py",          #Adding so AI has reference of this tool (so you can tell me what path you need)
            ],
            "10": [  # Custom Component Selection 🔍
                    # This is intentionally empty as paths will be dynamically added
                    # based on user selection in handle_component method
            ]
        }

        # ?
        self.path_map = {k: v + CORE_FILES for k, v in self.path_map.items()}  


    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')


    def get_context(self, context_type: List[str] = ["always"]):
        """Get context based on what's needed"""
        files_to_include = []
        for ctype in context_type:
            files_to_include.extend(self.CORE_FILES.get(ctype, []))
        return self.load_files(files_to_include)


    def create_context(self, 
                    include_types: List[str] = ["always"],
                    include_logs: bool = False) -> str:
        """Creates context with specified components"""
        context = []
        
        # Add core files based on types
        for ctype in include_types:
            files = self.CORE_FILES.get(ctype, [])
            for file in files:
                try:
                    content = self.load_file(file)
                    context.append(f"File: {file}\n{content}\n")
                except Exception as e:
                    context.append(f"Error loading {file}: {str(e)}\n")

        # Add logs if requested
        if include_logs:
            context.append(self.log_analyzer.get_recent_logs())

        # Ask about user context
        user_input = input("\nWould you like to include user context? (y/n): ").lower()
        if user_input == 'y':
            try:
                with open('context-master-scheme.md', 'r', encoding='utf-8') as f:
                    context.append(f.read())
            except Exception as e:
                context.append(f"Error loading user context: {str(e)}")

        return "\n".join(context)


    def append_user_context(self, output_file: str):
        """Append user context information to the output file"""
        try:
            with open('context-master-scheme.md', 'r', encoding='utf-8') as user_context:
                content = user_context.read()
                
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write("\n\n" + "=" * 50 + "\n")
                f.write("📚 ADDITIONAL CONTEXT\n")
                f.write("=" * 50 + "\n\n")
                f.write(content)
                
        except Exception as e:
            print(f"\033[91m❌ Error appending user context: {str(e)}\033[0m")


    def show_components(self, paths: List[str], section_name: str):
        """Show components and generate context"""
        # Clear screen and show logo first
        self.clear_screen()
        print(f"\033[96m{self.ascii_logo}\033[0m")
        
        context_file = os.path.join(os.getcwd(), f"context-user{section_name.lower()}.md")
        
        # Show initial status
        print("\n\033[96m📝 Generating context file...\033[0m")
        self.show_loading_animation()
        
        # Write header to file
        header = """🤖 ALGOBOT SYSTEM CONTEXT
    =========================

    📁 DIRECTORY STRUCTURE AND CONTENTS
    ================================"""
        
        try:
            with open(context_file, 'w', encoding='utf-8') as f:
                f.write(header)
                
                # Get and process tree output (write to file only, don't print)
                tree_output = self.get_clean_tree(paths)
                for line in tree_output:
                    f.write(f"{line}\n")
                
                # Add component relationships to file
                if paths:
                    first_component = paths[0].split('/')[1]
                    relationships = self.get_component_relationships_text(first_component)
                    f.write("\n\n" + relationships)
                
                # Append user context
                self.append_user_context(context_file)
            
            # Show success message with cool animation
            print("\r", end="")
            success_message = "🎉 Context file generated successfully! 🎉"
            for i in range(len(success_message)):
                print(f"\033[92m{success_message[:i+1]}\033[0m", end="\r")
                time.sleep(0.02)
            print("\n")
            print(f"\033[92m📄 File saved to: {context_file}\033[0m\n")
            
            # Show options menu
            while True:
                choice = self.show_options_menu()
                if choice == "1":
                    self.show_file_browser(self.base_dir)
                elif choice == "2":
                    self.show_architecture_guide(section_name)
                elif choice == "3":
                    break
                else:
                    print("\033[91mInvalid choice. Please try again.\033[0m")
                    time.sleep(1)
        
        except Exception as e:
            print(f"\033[91m❌ Error generating context: {str(e)}\033[0m")


    def show_loading_animation(self):
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        for _ in range(10):
            for frame in frames:
                print(f"\r\033[96m{frame} Loading AlgoBot Explorer...\033[0m", end="")
                time.sleep(0.1)
        print("\n")


    @classmethod
    def start(cls):
        """Main entry point for the explorer"""
        instance = cls()
        try:
            print("\033[96m🚀 Launching AlgoBot Explorer...\033[0m")
            time.sleep(1)
            instance.show_loading_animation()
            instance.calculate_project_stats()
            
            while True:
                instance.show_menu()
                print("\n\033[93m📊 Project Stats:\033[0m")
                print(f"   • Python Files: {instance.python_files}")
                print(f"   • Total Lines: {instance.total_lines}")
                choice = input("\n\033[92mChoose your component (1-10, Q to exit):\033[0m ").strip().upper()
                
                if choice == 'Q':
                    print("\033[96m👋 Happy trading!\033[0m")
                    break
                
                if choice in [str(i) for i in range(1, 11)]:
                    instance.handle_component(choice)
                else:
                    print("\033[91mInvalid selection!\033[0m")
        except Exception as e:
            print(f"\033[91m❌ Error in explorer: {str(e)}\033[0m")
            import traceback
            print(traceback.format_exc())

        except Exception as e:
            print(f"\033[91m❌ Error processing component: {str(e)}\033[0m")
            import traceback
            print("\033[93mDetailed error:\033[0m")
            print(traceback.format_exc())
            time.sleep(2)

    def calculate_project_stats(self):
        """Calculate project statistics with error handling"""
        try:
            self.python_files = 0
            self.total_lines = 0
            
            for root, _, files in os.walk(self.base_dir):
                if not any(glob.fnmatch.fnmatch(root, pat) for pat in self.exclude_patterns):
                    for file in files:
                        if file.endswith('.py'):
                            try:
                                file_path = os.path.join(root, file)
                                if not any(glob.fnmatch.fnmatch(file_path, pat) for pat in self.exclude_patterns):
                                    self.python_files += 1
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        self.total_lines += sum(1 for _ in f)
                            except Exception as e:
                                print(f"\033[93m⚠️ Warning: Could not process {file}: {str(e)}\033[0m")
        except Exception as e:
            print(f"\033[91m❌ Error calculating project stats: {str(e)}\033[0m")
            self.python_files = 0
            self.total_lines = 0

    

    def show_file_preview(self, file_path: str):
        """Show file preview with line numbers and syntax highlighting"""
        if os.path.exists(file_path):
            print(f"\n\033[95m📄 File Preview: {file_path}\033[0m")
            print("----------------------------------------")
            try:
                file_info = self.get_file_info(file_path)
                if file_info:
                    content = ''.join(file_info['content'])
                    
                    # Basic syntax highlighting for Python
                    import re
                    keywords = r'\b(def|class|import|from|return|if|else|elif|try|except|finally|with|as|for|while)\b'
                    content = re.sub(keywords, '\033[93m\\1\033[0m', content)  # Keywords in yellow
                    content = re.sub(r'(#.*$)', '\033[32m\\1\033[0m', content, flags=re.M)  # Comments in green
                    content = re.sub(r'"([^"]*)"', '\033[36m"\\1"\033[0m', content)  # Strings in cyan
                    
                    # Print with line numbers
                    for line_num, line in zip(file_info['line_numbers'], content.splitlines()):
                        print(f"\033[90m{line_num:4d}\033[0m │ {line}")
                    
                    print(f"\n\033[93mTotal lines: {file_info['line_count']}\033[0m")
            except Exception as e:
                print(f"\033[91mError reading file: {e}\033[0m")
            print("----------------------------------------\n")
    

    def show_file_browser(self, path: str):
        """Interactive file browser with arrow key navigation"""
        items = [item for item in os.listdir(path) 
                if not any(glob.fnmatch.fnmatch(item, pat) for pat in self.exclude_patterns)]
        selected = 0
        
        while True:
            self.clear_screen()
            print(f"\033[96m🗂️ File Browser: {path}\033[0m")
            print("----------------------------------------")
            
            for i, item in enumerate(items):
                full_path = os.path.join(path, item)
                if i == selected:
                    prefix = "► "
                else:
                    prefix = "  "
                    
                if os.path.isdir(full_path):
                    print(f"{prefix}\033[94m📁 {item}\033[0m")
                else:
                    print(f"{prefix}\033[37m📄 {item}\033[0m")
            
            try:
                import msvcrt
                key = ord(msvcrt.getch())
                if key == 27:  # Escape
                    break
                elif key == 13:  # Enter
                    full_path = os.path.join(path, items[selected])
                    if os.path.isdir(full_path):
                        self.show_file_browser(full_path)
                    else:
                        self.show_file_preview(full_path)
                        input("\nPress Enter to continue...")
                elif key == 72:  # Up arrow
                    selected = max(0, selected - 1)
                elif key == 80:  # Down arrow
                    selected = min(len(items) - 1, selected + 1)
            except ImportError:
                # Fallback for non-Windows systems
                cmd = input("\nEnter command (u/d/enter/q): ").lower()
                if cmd == 'q':
                    break
                elif cmd == 'u':
                    selected = max(0, selected - 1)
                elif cmd == 'd':
                    selected = min(len(items) - 1, selected + 1)
                elif cmd == 'enter':
                    full_path = os.path.join(path, items[selected])
                    if os.path.isdir(full_path):
                        self.show_file_browser(full_path)
                    else:
                        self.show_file_preview(full_path)
                        input("\nPress Enter to continue...")
        
        print("\n\033[93m📊 Project Stats:\033[0m")
        print(f"   • Python Files: {self.python_files}")
        print(f"   • Total Lines: {self.total_lines}")


    def show_options_menu(self):
        """Display options menu with logo"""
        self.clear_screen()
        print(f"\033[96m{self.ascii_logo}\033[0m")
        print("\033[96m=== Options ===\033[0m")
        print("1. 📂 Browse Project Files")
        print("2. 📚 View System Architecture Guide")
        print("3. ✨ Exit")
        return input("\n\033[93mChoose an option (1-3):\033[0m ").strip()


    def show_menu(self):
        """Display the main menu with ASCII art"""
        print(self.ascii_logo)
        print("\033[96m================================")
        print("1. 🎯 Core Infrastructure 🎯")
        print("   Event Bus | Database | Core Systems")
        print("2. 🔄 Data Pipeline & Storage 🔄")
        print("   Validation | Repository | Processing | Sources | Analysis")
        print("3. 📈 Trading Engine 📈 ")
        print("   Strategies | Backtesting | Execution")
        print("4. 🔬 Analysis Engine 🔬")
        print("   Market Data | Time Series | Signals")
        print("5. 🎨 UI & Dashboard 🎨")
        print("   Components | Pages | Monitoring")
        print("6. ⚙️ System Configuration ⚙️")
        print("   Environment | Docker | Settings")
        print("7. 🛠️ Development Tools 🛠️")
        print("   Tests | Scripts | Documentation")
        print("8. 🚀 Runtime Core 🚀")
        print("   Bot | Events | Integration")
        print("9. 🧪 Test Structure 🧪")
        print("   Core | Data | Integration Tests")
        print("10. 🔍 Custom Component Selection 🔍")
        print("================================\033[0m")

        print("\033[92mChoose your component (1-9, Q to exit):\033[0m")


    def generate_context(self, paths: List[str], section_name: str):
        context_file = os.path.join(os.getcwd(), f"ai_context_{section_name.lower()}.md")
        
        header = f"""🤖 ALGOBOT SYSTEM CONTEXT - {section_name}
    =========================
    📁 DIRECTORY STRUCTURE AND CONTENTS
    ================================"""
        
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write(header)
            
            # Write main content
            for path in paths:
                if os.path.exists(path):
                    f.write(f"\n=== {path} ===\n")
                    if os.path.isdir(path):
                        for root, _, files in os.walk(path):
                            if not any(glob.fnmatch.fnmatch(root, pat) for pat in self.exclude_patterns):
                                for file in sorted(files):
                                    if file.endswith(('.py', '.json', '.yml', '.yaml', '.md')):
                                        file_path = os.path.join(root, file)
                                        rel_path = os.path.relpath(file_path, path)
                                        f.write(f"│   {rel_path}\n")
                                        if file.endswith('.py'):
                                            try:
                                                with open(file_path, 'r', encoding='utf-8') as src:
                                                    content = src.read().strip()
                                                    f.write(f"    {content}\n")
                                            except Exception as e:
                                                f.write(f"    Error reading file: {e}\n")
                    else:
                        f.write(f"- {os.path.basename(path)}\n")
                        try:
                            with open(path, 'r', encoding='utf-8') as src:
                                content = src.read().strip()
                                f.write(f"    {content}\n")
                        except Exception as e:
                            f.write(f"    Error reading file: {e}\n")
                else:
                    f.write(f"\n=== {path} (Not Found) ===\n")

            # Add environment information
            f.write("\n\n=== 🌍 Environment Configuration ===\n")
            env_files = {
                'development': '.env.development',
                'production': '.env.production',
                'test': '.env.test'
            }
            
            for env_name, env_file in env_files.items():
                f.write(f"\n--- {env_name.upper()} Environment ---\n")
                try:
                    if os.path.exists(env_file):
                        with open(env_file, 'r') as env:
                            content = env.read().strip()
                            f.write(f"File: {env_file}\n")
                            f.write(f"Content:\n{content}\n")
                    else:
                        f.write(f"⚠️ {env_file} not found\n")
                except Exception as e:
                    f.write(f"❌ Error reading {env_file}: {str(e)}\n")

            # Add Docker configuration
            f.write("\n\n=== 🐳 Docker Configuration ===\n")
            docker_file = "docker-compose.yml"
            try:
                if os.path.exists(docker_file):
                    with open(docker_file, 'r') as docker:
                        content = docker.read().strip()
                        f.write(f"File: {docker_file}\n")
                        f.write(f"Content:\n{content}\n")
                else:
                    f.write("⚠️ docker-compose.yml not found\n")
            except Exception as e:
                f.write(f"❌ Error reading docker-compose.yml: {str(e)}\n")


    def get_component_relationships(self, component: str):
        """Display relationships for a component"""
        if component in self.relationships:
            print("\n\033[95m📊 Component Dependencies:\033[0m")
            for key, values in self.relationships[component].items():
                print(f"\n  \033[93m{key}:\033[0m")
                for dep in values:
                    print(f"    \033[90m• {dep}\033[0m")

    def handle_component(self, choice: str):
        """Handle component selection and viewing"""
        try:
            paths = []
            section_name = ""
            
            if choice == "10":
                selections = input("Enter component numbers (e.g., 4,6,7): ")
                components = selections.strip().split(',')
                section_name = "Custom Selection"
                
                for num in components:
                    num = num.strip()
                    if num in self.path_map:
                        paths.extend(self.path_map[num])
                        if num in ["3", "4", "8"]:
                            paths.extend([
                                p for p in self.path_map["5"] 
                                if any(dep in p for dep in ["trading.py", "metrics.py"])
                            ])
            else:
                paths = self.path_map.get(choice, [])
                section_name = {
                    "1": "Core Infrastructure",
                    "2": "Data Flow/Pipeline/Storage",
                    "3": "Trading Engine",
                    "4": "Analysis Engine",
                    "5": "UI & Dashboard",
                    "6": "System Configuration",
                    "7": "Development Tools",
                    "8": "Runtime Core",
                    "9": "Tests",
                    "9": "Custom Selection(e.g., 4,6,7)",
                }.get(choice, "Unknown")
            
            if not paths:
                print("\033[91mNo paths found for selected component\033[0m")
                return
            
            # Show loading animation
            self.show_loading_animation()
            
            # Display structure and generate context
            self.display_structure(paths)
            self.show_components(paths, section_name)
            
        except Exception as e:
            print(f"\033[91m❌ Error processing component: {e}\033[0m")
            time.sleep(2)

    
    def get_clean_tree(self, paths: List[str]) -> List[str]:
        """Generate clean tree structure for paths"""
        output = []
        
        for path in paths:
            if os.path.exists(path):
                if os.path.isdir(path):
                    output.append(f"\n=== {path} (Directory) ===")
                    for root, _, files in os.walk(path):
                        if not any(glob.fnmatch.fnmatch(root, pat) for pat in self.exclude_patterns):
                            for file in sorted(files):
                                if file.endswith('.py') and not file.startswith('__'):
                                    rel_path = os.path.relpath(os.path.join(root, file), path)
                                    output.append(f"│   {rel_path}")
                else:
                    output.append(f"\n=== {path} (File) ===")
                    output.append(f"- {os.path.basename(path)}")
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            output.extend([f"    {line}" for line in content.split('\n')])
                    except Exception as e:
                        output.append(f"    Error reading file: {e}")
            else:
                output.append(f"\n=== {path} (Not Found) ===")
        
        return output
    


    def get_file_info(self, file_path: str) -> Dict[str, any]:
        """Get file information including line numbers and content"""
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
            return None
    


    def get_line_numbers(self, file_path: str, search_term: str = None) -> List[int]:
        """Get line numbers for a file, optionally filtering by search term"""
        try:
            file_info = self.get_file_info(file_path)
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
        


    def get_file_section(self, file_path: str, start_line: int, end_line: int) -> str:
        """Get a specific section of a file by line numbers"""
        try:
            file_info = self.get_file_info(file_path)
            if not file_info:
                return ""
                
            lines = file_info['content']
            return ''.join(lines[start_line-1:end_line])
        except Exception as e:
            print(f"\033[91mError getting file section: {e}\033[0m")
            return ""

    def get_env_files(self):
        """Get environment file contents"""
        env_files = {
            'development': '.env.development',
            'production': '.env.production',
            'test': '.env.test'
        }
        
        env_contents = {}
        for env_name, env_file in env_files.items():
            try:
                with open(env_file, 'r') as f:
                    env_contents[env_name] = f.read()
            except Exception as e:
                env_contents[env_name] = f"Error reading {env_file}: {str(e)}"
                
        return env_contents
    

if __name__ == "__main__":
    explorer = ContextCreator()
    explorer.start()





# Example line - counting usage:
# line_numbers = explorer.get_line_numbers("path/to/file.py", "def")  # Get all function definitions
# section = explorer.get_file_section("path/to/file.py", 10, 20)  # Get lines 10-20






# Static directory search if I ever decide to use this for whatever reason

    # def display_structure(self, paths: List[str]):
    #     """Display structured tree view of components"""
    #     print("\n📁 ALGOBOT STRUCTURE")
    #     print("==================")
        
    #     # Create structured tree
    #     tree = {
    #         "src": {
    #             "backtesting": {"files": []},
    #             "brokers": {"files": []},
    #             "core": {
    #                 "compression": {"files": []},
    #                 "mixins": {"files": []},
    #                 "files": []
    #             },
    #             "data": {
    #                 "sources": {"files": []},
    #                 "schemas": {"files": []},
    #                 "processors": {"files": []},
    #                 "files": []
    #             },
    #             "strategies": {
    #                 "alpha": {"files": []},
    #                 "risk": {"files": []},
    #                 "indicators": {"files": []},
    #                 "files": []
    #             },
    #             "engines": {
    #                 "analysis": {"files": []},
    #                 "execution": {"files": []},
    #                 "pre_analysis": {"files": []},
    #                 "watchlist": {"files": []}
    #             },
    #             "ui": {
    #                 "components": {"files": []},
    #                 "pages": {"files": []},
    #                 "layouts": {"files": []},
    #                 "styles": {"files": []},
    #                 "utils": {"files": []},
    #                 "files": []
    #             },
    #             "config": {"files": []}
    #         },
    #         "tests": {
    #             "core": {"files": []},
    #             "data": {"files": []},
    #             "infrastructure": {"files": []},
    #             "integration": {"files": []},
    #             "trading": {"files": []},
    #             "utils": {"files": []},
    #             "files": []
    #         },
    #         "config": {
    #             "files": []
    #         },
    #         "scripts": {
    #             "files": []
    #         }}
       
    #     for path in paths:   # Populate tree with files
    #         parts = path.split('/')
    #         current = tree
    #         for part in parts[:-1]:
    #             if part not in current:
    #                 current[part] = {"files": []}
    #             current = current[part]
    #         current["files"].append(parts[-1])

    #     def print_tree(structure, prefix=""):
    #         """Inner function to print tree structure"""
    #         for key, value in sorted(structure.items()):
    #             if key != "files":
    #                 print(f"{prefix}├── {key}")
    #                 if "files" in value and value["files"]:
    #                     for file in sorted(value["files"]):
    #                         print(f"{prefix}│   ├── {file}")
    #                 print_tree(value, prefix + "│   ")        

    #     print_tree(tree)  
    #     print("\n")