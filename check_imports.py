"""
Installation Verification Script for License Plate Detection Project
Checks all required dependencies and their functionality
"""

import sys
import importlib
import subprocess
from datetime import datetime

try:
    from packaging.version import parse as version_parse
except ImportError:
    print("Installing 'packaging' module for version comparison...")
    subprocess.run([sys.executable, "-m", "pip", "install", "packaging"])
    from packaging.version import parse as version_parse

class InstallationChecker:
    def __init__(self):
        self.results = {}
        self.failed_packages = []
        self.outdated_packages = []
        self.success_count = 0
        self.total_count = 0
        # Define required packages
        self.required_packages = {
            'ultralytics': {'import_name': 'ultralytics', 'min_version': '8.1.0'},
            'opencv-python': {'import_name': 'cv2', 'min_version': '4.8.0'},
            'torch': {'import_name': 'torch', 'min_version': '2.0.0'},
            'torchvision': {'import_name': 'torchvision', 'min_version': '0.15.0'},
            'numpy': {'import_name': 'numpy', 'min_version': '1.24.0'},
            'Pillow': {'import_name': 'PIL', 'min_version': '10.0.0'},
            'matplotlib': {'import_name': 'matplotlib', 'min_version': '3.7.0'},
            'requests': {'import_name': 'requests', 'min_version': '2.31.0'},
            'tqdm': {'import_name': 'tqdm', 'min_version': '4.65.0'},
            'pytesseract': {'import_name': 'pytesseract', 'min_version': '0.3.10'}
        }

    def print_header(self):
        print("🔍 License Plate Detection Project - Installation Verification")
        print("=" * 65)
        print(f"📅 Check Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🐍 Python Version: {sys.version}")
        print("=" * 65)

    def check_package_installation(self, package_name, import_name, min_version):
        try:
            module = importlib.import_module(import_name)
            version = self.get_package_version(module, import_name)
            if version != 'Version info not available' and version_parse(version) < version_parse(min_version):
                return False, f"{version} (Below min required {min_version})", 'outdated'
            return True, version, None
        except ImportError as e:
            return False, str(e), 'missing'
        except Exception as e:
            return False, f"Error: {str(e)}", 'error'

    def get_package_version(self, module, import_name):
        version_attrs = ['__version__', 'version', 'VERSION']
        for attr in version_attrs:
            if hasattr(module, attr):
                version = getattr(module, attr)
                return str(version) if not callable(version) else str(version())
        return 'Version info not available'

    def check_all_packages(self):
        print("\n📦 Checking Package Installations...")
        print("-" * 50)
        for package_name, config in self.required_packages.items():
            import_name = config['import_name']
            min_version = config['min_version']
            self.total_count += 1
            is_installed, version_or_error, status = self.check_package_installation(package_name, import_name, min_version)
            if is_installed:
                print(f"✅ {package_name:<15} v{version_or_error}")
                self.success_count += 1
            elif status == 'outdated':
                print(f"⚠️  {package_name:<15} INSTALLED but outdated: v{version_or_error}")
                self.outdated_packages.append(package_name)
            else:
                print(f"❌ {package_name:<15} NOT INSTALLED - {version_or_error}")
                self.failed_packages.append(package_name)

    def check_tesseract_binary(self):
        print("\n🧐 Checking Tesseract system OCR binary...")
        from shutil import which
        tesseract_path = which("tesseract")
        if tesseract_path:
            print(f"✅ Tesseract binary found at: {tesseract_path}")
            try:
                version = subprocess.check_output(["tesseract", "--version"], stderr=subprocess.STDOUT, text=True)
                print(f"   Version info: {version.splitlines()[0]}")
            except Exception as e:
                print(f"   Error getting version info: {e}")
        else:
            print("❌ Tesseract binary is NOT found in your PATH!")
            print("   OCR will not work until you install and add Tesseract to your system PATH.")
            if sys.platform.startswith("win"):
                print("   Windows users: Download from https://github.com/UB-Mannheim/tesseract/wiki")
            else:
                print("   Use your system's package manager to install (e.g., `apt install tesseract-ocr`)")

    def print_summary(self):
        print("\n" + "=" * 65)
        print("📊 INSTALLATION SUMMARY")
        print("=" * 65)
        total_missing = len(self.failed_packages)
        total_outdated = len(self.outdated_packages)
        ok = self.success_count
        total = self.total_count
        succ_rate = (ok / total) * 100 if total > 0 else 0
        print(f"Total Packages Checked: {total}")
        print(f"Successfully Installed: {ok}")
        print(f"Outdated Packages: {total_outdated}")
        print(f"Missing Packages: {total_missing}")
        print(f"Success Rate: {succ_rate:.1f}%")
        if self.outdated_packages:
            print(f"\n⚠️  OUTDATED PACKAGES:")
            for package in self.outdated_packages:
                print(f"   - {package}")
        if self.failed_packages:
            print(f"\n❌ MISSING PACKAGES:")
            for package in self.failed_packages:
                print(f"   - {package}")
            print(f"\n🔧 TO FIX, RUN:")
            print(f"pip install {' '.join(self.failed_packages + self.outdated_packages)}")
        print("\n📖 For system Tesseract install info, always see your README.")

    def run_complete_check(self):
        self.print_header()
        self.check_all_packages()
        self.check_tesseract_binary()
        self.print_summary()

def main():
    checker = InstallationChecker()
    checker.run_complete_check()

if __name__ == "__main__":
    main()
