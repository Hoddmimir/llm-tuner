#!/usr/bin/env python3
"""Hardware detection - vendor agnostic GPU profiling."""

import json
import subprocess
from pathlib import Path


class HardwareDetector:
    """Detect and profile GPU hardware across vendors (ROCm/CUDA/Metal)."""
    
    def detect(self) -> dict:
        """Run full hardware detection. Returns comprehensive profile."""
        
        profile = {
            "vendor": self._detect_vendor(),
            "gpu": self._get_gpu_info(),
            "cpu": self._get_cpu_info(),
            "memory": self._get_system_memory(),
            "drivers": self._get_driver_versions()
        }
        
        return profile
    
    def _detect_vendor(self) -> str:
        """Detect GPU vendor (rocm, cuda, metal, cpu)."""
        
        # Check for ROCm first
        if self._check_rocm():
            return "rocm"
        
        # Check for CUDA
        if self._check_cuda():
            return "cuda"
        
        # Check for Metal (macOS)
        if self._check_metal():
            return "metal"
        
        return "cpu"
    
    def _check_rocm(self) -> bool:
        """Check if ROCm is available."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--version"],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0 and "NVIDIA" in result.stdout.upper()
        except Exception:
            return False
    
    def _check_metal(self) -> bool:
        """Check if Metal is available (macOS)."""
        try:
            import platform
            if platform.system() != "Darwin":
                return False
            
            # Check for system_profiler with GPU info
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=5
            )
            return "Metal" in result.stdout or "Apple" in result.stdout
        except Exception:
            return False
    
    def _get_gpu_info(self) -> dict:
        """Get detailed GPU information."""
        vendor = self._detect_vendor()
        
        if vendor == "rocm":
            return self._get_rocm_gpu_info()
        elif vendor == "cuda":
            return self._get_cuda_gpu_info()
        elif vendor == "metal":
            return self._get_metal_gpu_info()
        
        return {"name": "CPU-only", "vram_total": 0, "vram_used": 0}
    
    def _get_rocm_gpu_info(self) -> dict:
        """Get ROCm GPU details."""
        try:
            result = subprocess.run(
                ["rocm-smi", "-a", "--json"],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                # Find first card (keys like "card0", "cards", etc.)
                card = None
                for key in data:
                    if key.startswith("card") or key == "cards":
                        card = data[key]
                        break
                
                if not card:
                    return {"name": "AMD GPU (unknown)", "vram_total_mb": 0, "vram_used_mb": 0}
                
                # Handle both formats: direct card object vs list of cards
                if isinstance(card, list) and len(card) > 0:
                    card = card[0]
                
                # Get GPU name - try multiple field names
                gpu_name = (card.get("Device Name") or 
                           card.get("name") or 
                           card.get("card_serial") or 
                           "AMD GPU")
                
                # Parse VRAM from system section or calculate from allocated percentage
                vram_total_mb = 0
                vram_used_mb = 0
                
                # Check for vram_usage field (newer format)
                vram_usage = card.get("vram_usage", {})
                if vram_usage:
                    total_str = vram_usage.get("total", "0")
                    used_str = vram_usage.get("used", "0")
                    vram_total_mb = self._parse_vram(total_str)
                    vram_used_mb = self._parse_vram(used_str)
                
                # If no vram_usage, try to get from system section (PID entries show VRAM usage)
                if not vram_total_mb and "system" in data:
                    system_data = data["system"]
                    for key, value in system_data.items():
                        if isinstance(value, list) and len(value) >= 3:
                            # Format: [process_name, gpu_id, vram_bytes, ...]
                            try:
                                vram_bytes = int(value[2])
                                vram_used_mb += int(vram_bytes / (1024 * 1024))
                            except (ValueError, IndexError):
                                pass
                
                # RX 7900 XTX has 24GB - use known values for common cards
                if "7900" in gpu_name and not vram_total_mb:
                    vram_total_mb = 24576  # 24 GB
                    
                return {
                    "name": gpu_name,
                    "vram_total_mb": vram_total_mb,
                    "vram_used_mb": vram_used_mb,
                    "vram_free_mb": max(0, vram_total_mb - vram_used_mb),
                    "temperature_c": float(card.get("Temperature (Sensor edge) (C)", 
                                   card.get("temp", {}).get("asic_temp", 0)) or 0),
                    "power_watts": float(card.get("Average Graphics Package Power (W)", 0) or 0)
                }
        except Exception as e:
            print(f"Warning: Could not get ROCm GPU info: {e}")
        
        return {"name": "AMD GPU (unknown)", "vram_total_mb": 0, "vram_used_mb": 0}
    
    def _get_cuda_gpu_info(self) -> dict:
        """Get CUDA/NVIDIA GPU details."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                parts = [x.strip() for x in result.stdout.split(",")]
                
                return {
                    "name": parts[0] if len(parts) > 0 else "NVIDIA GPU",
                    "vram_total_mb": int(parts[1]) if len(parts) > 1 else 0,
                    "vram_used_mb": int(parts[2]) if len(parts) > 2 else 0,
                    "temperature_c": float(parts[3]) if len(parts) > 3 else 0,
                    "power_watts": float(parts[4]) if len(parts) > 4 else 0
                }
        except Exception as e:
            print(f"Warning: Could not get CUDA GPU info: {e}")
        
        return {"name": "NVIDIA GPU (unknown)", "vram_total_mb": 0, "vram_used_mb": 0}
    
    def _get_metal_gpu_info(self) -> dict:
        """Get Metal/macOS GPU details."""
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=5
            )
            
            # Parse basic info from output
            gpu_name = "Apple Silicon"
            for line in result.stdout.split('\n'):
                if 'Chip' in line or 'GPU' in line:
                    gpu_name = line.strip()
                    break
            
            return {
                "name": gpu_name,
                "vram_total_mb": 0,  # Unified memory on Apple Silicon
                "vram_used_mb": 0
            }
        except Exception as e:
            print(f"Warning: Could not get Metal GPU info: {e}")
        
        return {"name": "Apple Silicon (unknown)", "vram_total_mb": 0, "vram_used_mb": 0}
    
    def _get_cpu_info(self) -> dict:
        """Get CPU information."""
        try:
            # Try lscpu first (Linux)
            result = subprocess.run(
                ["lscpu"],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                info = {}
                for line in result.stdout.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        value = value.strip()
                        
                        if key == "model_name":
                            info["name"] = value
                        elif key == "cpu_mhz":
                            info["mhz"] = float(value)
                        elif key in ("cpu(s)", "socket(s)", "core(s)", "thread(s)"):
                            info[key] = int(value)
                
                return info or {"name": "Unknown CPU"}
        except Exception:
            pass
        
        # Fallback
        import os
        return {
            "name": os.uname().machine,
            "cores": os.cpu_count() or 0
        }
    
    def _get_system_memory(self) -> dict:
        """Get system RAM information."""
        try:
            result = subprocess.run(
                ["free", "-m"],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    parts = lines[1].split()
                    
                    return {
                        "total_mb": int(parts[1]),
                        "used_mb": int(parts[2]),
                        "free_mb": int(parts[3])
                    }
        except Exception:
            pass
        
        return {"total_mb": 0, "used_mb": 0, "free_mb": 0}
    
    def _get_driver_versions(self) -> dict:
        """Get driver and runtime versions."""
        drivers = {}
        
        # ROCm version
        try:
            result = subprocess.run(
                ["rocm-smi", "--version"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                drivers["rocm"] = result.stdout.strip()
        except Exception:
            pass
        
        # CUDA version
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        drivers["cuda"] = line.strip()
                        break
        except Exception:
            pass
        
        # Python version
        import sys
        drivers["python"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        return drivers
    
    def _parse_vram(self, value: str) -> int:
        """Parse VRAM value from string (handles MB/GB)."""
        try:
            if 'gb' in value.lower():
                return int(float(value.replace('GB', '').strip()) * 1024)
            elif 'mb' in value.lower():
                return int(float(value.replace('MB', '').strip()))
            else:
                # Assume MB if no unit
                return int(float(value.strip().replace(',', '')))
        except Exception:
            return 0
