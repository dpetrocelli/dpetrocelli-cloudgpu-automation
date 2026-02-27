"""
Vast.ai GPU Instance Manager.

A Python module for creating and managing GPU instances on Vast.ai.
Follows PEP 8 style guide and PEP 257 docstring conventions.

Example:
    >>> from vast_gpu_manager import VastGPUManager
    >>> manager = VastGPUManager()
    >>> offers = manager.search_gpus(gpu_name="RTX_4090", max_price=0.50)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vastai_sdk import VastAI

# Configure module logger
logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
        logger.debug("Loaded environment from %s", _env_path)
except ImportError:
    logger.debug("python-dotenv not installed, using system environment variables")


# Constants
DEFAULT_GPU_NAME = "RTX_4090"
DEFAULT_NUM_GPUS = 1
DEFAULT_MAX_PRICE = 0.50
DEFAULT_MIN_RELIABILITY = 0.95
DEFAULT_DOCKER_IMAGE = "vastai/pytorch"
DEFAULT_DISK_SPACE = 20.0

# Common Docker images for ML/AI workloads
# vastai/* images are pre-cached and start faster
DOCKER_IMAGES: dict[str, str] = {
    "pytorch": "vastai/pytorch",
    "tensorflow": "vastai/tensorflow",
    "cuda": "nvidia/cuda:12.1.0-runtime-ubuntu22.04",
    "cuda_devel": "nvidia/cuda:12.1.0-devel-ubuntu22.04",
    # Alternative images (not cached, slower to start)
    "pytorch_official": "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    "huggingface": "huggingface/transformers-pytorch-gpu:latest",
}


def get_env(key: str, default: str | None = None) -> str | None:
    """
    Get environment variable with optional default value.

    Args:
        key: Environment variable name.
        default: Default value if variable is not set.

    Returns:
        Environment variable value or default.
    """
    return os.environ.get(key, default)


def get_env_float(key: str, default: float) -> float:
    """
    Get environment variable as float.

    Args:
        key: Environment variable name.
        default: Default value if variable is not set or invalid.

    Returns:
        Float value from environment or default.
    """
    try:
        value = os.environ.get(key)
        return float(value) if value else default
    except ValueError:
        logger.warning("Invalid float value for %s, using default: %s", key, default)
        return default


def extract_gpu_model_number(gpu_name: str) -> int | None:
    """
    Extract the numeric model number from a GPU name.

    Args:
        gpu_name: GPU name like "RTX 3090", "GTX 1080 Ti", "A100".

    Returns:
        The extracted model number or None if not found.

    Examples:
        "RTX 3090" -> 3090
        "GTX 1080 Ti" -> 1080
        "A100" -> 100
        "H100" -> 100
        "Tesla V100" -> 100
    """
    import re

    # Find all numbers in the GPU name
    numbers = re.findall(r"\d+", gpu_name)
    if numbers:
        # Return the largest number (usually the model number)
        return max(int(n) for n in numbers)
    return None


def get_env_int(key: str, default: int) -> int:
    """
    Get environment variable as integer.

    Args:
        key: Environment variable name.
        default: Default value if variable is not set or invalid.

    Returns:
        Integer value from environment or default.
    """
    try:
        value = os.environ.get(key)
        return int(value) if value else default
    except ValueError:
        logger.warning("Invalid int value for %s, using default: %s", key, default)
        return default


@dataclass
class GPUConfig:
    """
    Configuration for GPU search and instance creation.

    Attributes:
        gpu_name: GPU model name (e.g., "RTX_4090", "A100").
        num_gpus: Number of GPUs required.
        max_price: Maximum price per hour in USD.
        min_reliability: Minimum reliability score (0-1).
        docker_image: Docker image to use.
        disk_space: Disk space in GB.
    """

    gpu_name: str = field(default_factory=lambda: get_env("DEFAULT_GPU_NAME", DEFAULT_GPU_NAME))
    num_gpus: int = field(default_factory=lambda: get_env_int("DEFAULT_NUM_GPUS", DEFAULT_NUM_GPUS))
    max_price: float = field(
        default_factory=lambda: get_env_float("DEFAULT_MAX_PRICE", DEFAULT_MAX_PRICE)
    )
    min_reliability: float = field(
        default_factory=lambda: get_env_float("DEFAULT_MIN_RELIABILITY", DEFAULT_MIN_RELIABILITY)
    )
    docker_image: str = field(
        default_factory=lambda: get_env("DEFAULT_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE)
    )
    disk_space: float = DEFAULT_DISK_SPACE


@dataclass(frozen=True)
class GPUOffer:
    """
    Represents an available GPU offer on Vast.ai.

    Attributes:
        id: Unique offer identifier.
        gpu_name: GPU model name.
        num_gpus: Number of GPUs in the offer.
        gpu_ram: GPU RAM in GB.
        cpu_cores: Number of CPU cores.
        ram: System RAM in GB.
        disk_space: Available disk space in GB.
        price_per_hour: Price per hour in USD.
        reliability: Reliability score (0-1).
        location: Geographic location.
    """

    id: int
    gpu_name: str
    num_gpus: int
    gpu_ram: float
    cpu_cores: int
    ram: float
    disk_space: float
    price_per_hour: float
    reliability: float
    location: str = ""

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> GPUOffer:
        """
        Create GPUOffer from Vast.ai API response.

        Args:
            data: Raw API response dictionary.

        Returns:
            GPUOffer instance.
        """
        return cls(
            id=data.get("id", 0),
            gpu_name=data.get("gpu_name", "Unknown"),
            num_gpus=data.get("num_gpus", 0),
            gpu_ram=data.get("gpu_ram", 0.0),
            cpu_cores=data.get("cpu_cores", 0),
            ram=data.get("cpu_ram", 0.0),
            disk_space=data.get("disk_space", 0.0),
            price_per_hour=data.get("dph_total", 0.0),
            reliability=data.get("reliability", 0.0),
            location=data.get("geolocation", ""),
        )


@dataclass
class Instance:
    """
    Represents a running GPU instance.

    Attributes:
        id: Instance identifier.
        gpu_name: GPU model name.
        num_gpus: Number of GPUs.
        status: Current instance status.
        ssh_host: SSH hostname.
        ssh_port: SSH port number.
        jupyter_url: Jupyter notebook URL if enabled.
    """

    id: int
    gpu_name: str
    num_gpus: int
    status: str
    ssh_host: str | None = None
    ssh_port: int | None = None
    jupyter_url: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> Instance:
        """
        Create Instance from Vast.ai API response.

        Args:
            data: Raw API response dictionary.

        Returns:
            Instance object.
        """
        return cls(
            id=data.get("id", 0),
            gpu_name=data.get("gpu_name", "Unknown"),
            num_gpus=data.get("num_gpus", 0),
            status=data.get("actual_status", "unknown"),
            ssh_host=data.get("ssh_host"),
            ssh_port=data.get("ssh_port"),
            jupyter_url=data.get("jupyter_url"),
        )

    @property
    def ssh_command(self) -> str | None:
        """
        Get SSH command to connect to this instance.

        Returns:
            SSH command string or None if SSH not available.
        """
        if self.ssh_host and self.ssh_port:
            return f"ssh -p {self.ssh_port} root@{self.ssh_host}"
        return None


class VastAIError(Exception):
    """Base exception for Vast.ai operations."""

    pass


class APIKeyError(VastAIError):
    """Raised when API key is missing or invalid."""

    pass


class InstanceNotFoundError(VastAIError):
    """Raised when instance is not found."""

    pass


class VastGPUManager:
    """
    Manager class for Vast.ai GPU instances.

    This class provides a high-level interface for searching, creating,
    and managing GPU instances on the Vast.ai platform.

    Attributes:
        config: Default configuration for GPU operations.

    Example:
        >>> manager = VastGPUManager(api_key="your_api_key")
        >>> offers = manager.search_gpus(gpu_name="RTX_3090", max_price=0.5)
        >>> instance = manager.launch_instance(gpu_name="RTX_3090")
        >>> manager.stop_instance(instance_id=12345)
    """

    def __init__(
        self,
        api_key: str | None = None,
        config: GPUConfig | None = None,
    ) -> None:
        """
        Initialize the Vast.ai GPU Manager.

        Args:
            api_key: Vast.ai API key. If not provided, will look for
                VASTAI_API_KEY environment variable or .env file.
            config: Default GPU configuration. If not provided, uses
                values from environment or defaults.

        Raises:
            APIKeyError: If no API key is provided or found.
        """
        self.api_key = api_key or get_env("VASTAI_API_KEY")
        if not self.api_key:
            raise APIKeyError(
                "API key required. Pass api_key parameter, "
                "set VASTAI_API_KEY environment variable, "
                "or add it to .env file"
            )

        self.config = config or GPUConfig()
        self._sdk: VastAI | None = None

        logger.info("VastGPUManager initialized")

    @property
    def sdk(self) -> VastAI:
        """
        Lazy load the Vast.ai SDK.

        Returns:
            VastAI SDK instance.

        Raises:
            ImportError: If vastai-sdk is not installed.
        """
        if self._sdk is None:
            try:
                from vastai_sdk import VastAI

                self._sdk = VastAI(api_key=self.api_key)
                logger.debug("VastAI SDK initialized")
            except ImportError as e:
                raise ImportError("vastai-sdk not installed. Run: pip install vastai-sdk") from e
        return self._sdk

    def search_gpus(
        self,
        gpu_name: str | None = None,
        gpu_family: str | None = None,
        min_model: int | None = None,
        num_gpus: int | None = None,
        min_gpu_ram: float | None = None,
        min_cpu_ram: float | None = None,
        max_price: float | None = None,
        min_reliability: float | None = None,
        min_cuda: float | None = None,
        min_dlperf: float | None = None,
        min_tflops: float | None = None,
        limit: int = 20,
        use_defaults: bool = False,
        order_by: str = "price",
        order_desc: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Search for available GPU offers.

        Args:
            gpu_name: GPU model search term (e.g., "3090", "RTX_3090", "A100").
                Partial match: "3090" matches "RTX 3090", "RTX 3090 Ti", etc.
                If None and use_defaults=False, no GPU filter is applied.
            gpu_family: GPU family prefix (e.g., "RTX", "GTX", "Tesla", "A100").
                Filters GPUs that start with this prefix (case-insensitive).
            min_model: Minimum GPU model number (e.g., 3080 means 3080 and above).
                Extracts numeric part from GPU name and filters.
            num_gpus: Minimum number of GPUs required.
                If None, defaults to 1.
            min_gpu_ram: Minimum GPU RAM in GB. Optional filter.
            min_cpu_ram: Minimum system RAM in GB. Optional filter.
            max_price: Maximum price per hour in USD.
                If None, no price limit is applied.
            min_reliability: Minimum reliability score (0-1).
                If None, defaults to 0 (no filter).
            min_cuda: Minimum CUDA version (e.g., 12.0). Optional filter.
            min_dlperf: Minimum DLPerf score. Optional filter.
            min_tflops: Minimum TFLOPS compute power. Optional filter.
            limit: Maximum number of results to return.
            use_defaults: If True, use config defaults for unspecified params.
                If False (default), only apply filters you explicitly set.
            order_by: Field to sort by. Options: "price", "gpu_ram", "reliability",
                "num_gpus", "score", "dlperf" (deep learning performance).
            order_desc: Sort descending if True, ascending if False (default).

        Returns:
            List of available GPU offers as dictionaries.
        """
        # Only use config defaults if explicitly requested
        if use_defaults:
            gpu_name = gpu_name if gpu_name is not None else self.config.gpu_name
            num_gpus = num_gpus if num_gpus is not None else self.config.num_gpus
            max_price = max_price if max_price is not None else self.config.max_price
            min_reliability = (
                min_reliability if min_reliability is not None else self.config.min_reliability
            )
        else:
            # Use minimal defaults - don't restrict unless asked
            num_gpus = num_gpus if num_gpus is not None else 1
            min_reliability = min_reliability if min_reliability is not None else 0

        query_parts = [
            "rented=False",
            "rentable=True",
            f"num_gpus>={num_gpus}",
        ]

        # Only add reliability filter if > 0
        if min_reliability and min_reliability > 0:
            query_parts.append(f"reliability>={min_reliability}")

        # Note: gpu_name, gpu_ram and cpu_ram filters are applied locally after API call
        # gpu_name uses partial match (contains) for flexible searching

        # Only add price filter if specified
        if max_price is not None:
            query_parts.append(f"dph_total<={max_price}")

        query = " ".join(query_parts)
        logger.debug("Search query: %s", query)

        result = self.sdk.search_offers(query=query)

        if isinstance(result, list):
            # Filter by GPU name (partial match / contains)
            if gpu_name:
                gpu_name_upper = gpu_name.upper().replace("_", " ")
                result = [
                    r for r in result if gpu_name_upper in r.get("gpu_name", "").upper()
                ]

            # Filter by GPU family (prefix match)
            if gpu_family:
                family_upper = gpu_family.upper()
                result = [
                    r for r in result if r.get("gpu_name", "").upper().startswith(family_upper)
                ]

            # Filter by minimum model number
            if min_model is not None:
                filtered = []
                for r in result:
                    model_num = extract_gpu_model_number(r.get("gpu_name", ""))
                    if model_num is not None and model_num >= min_model:
                        filtered.append(r)
                result = filtered

            # Filter by GPU RAM (local filter - API doesn't support this)
            if min_gpu_ram is not None:
                min_gpu_ram_mb = min_gpu_ram * 1024  # Convert GB to MB
                result = [r for r in result if r.get("gpu_ram", 0) >= min_gpu_ram_mb]

            # Filter by CPU RAM (local filter - API doesn't support this)
            if min_cpu_ram is not None:
                min_cpu_ram_mb = min_cpu_ram * 1024  # Convert GB to MB
                result = [r for r in result if r.get("cpu_ram", 0) >= min_cpu_ram_mb]

            # Filter by minimum CUDA version
            if min_cuda is not None:
                result = [r for r in result if r.get("cuda_max_good", 0) >= min_cuda]

            # Filter by minimum DLPerf (deep learning performance)
            if min_dlperf is not None:
                result = [r for r in result if r.get("dlperf", 0) >= min_dlperf]

            # Filter by minimum TFLOPS
            if min_tflops is not None:
                result = [r for r in result if r.get("total_flops", 0) >= min_tflops]

            # Calculate value (DLPerf per dollar) for each offer
            for r in result:
                price = r.get("dph_total", 0)
                dlperf = r.get("dlperf", 0)
                r["dlperf_per_dollar"] = dlperf / price if price > 0 else 0
                # Also calculate TFLOPS per dollar
                tflops = r.get("total_flops", 0)
                r["tflops_per_dollar"] = tflops / price if price > 0 else 0

            # Sort results
            sort_key_map = {
                "price": "dph_total",
                "gpu_ram": "gpu_ram",
                "reliability": "reliability",
                "num_gpus": "num_gpus",
                "score": "score",
                "dlperf": "dlperf",
                "cpu_ram": "cpu_ram",
                "disk_space": "disk_space",
                "tflops": "total_flops",
                "cuda": "cuda_max_good",
                "value": "dlperf_per_dollar",
                "tflops_value": "tflops_per_dollar",
                "price_power": "dph_total",  # Special: price asc, then dlperf desc
            }
            sort_field = sort_key_map.get(order_by, "dph_total")

            if order_by == "price_power":
                # Multi-key sort: price ascending, then dlperf descending
                sorted_result = sorted(
                    result,
                    key=lambda x: (x.get("dph_total", 0) or 0, -(x.get("dlperf", 0) or 0)),
                )
            else:
                sorted_result = sorted(
                    result,
                    key=lambda x: x.get(sort_field, 0) or 0,
                    reverse=order_desc,
                )

            logger.info("Found %d offers, sorted by %s", len(sorted_result[:limit]), order_by)
            return sorted_result[:limit]

        return result

    def launch_instance(
        self,
        gpu_name: str | None = None,
        image: str | None = None,
        num_gpus: int | None = None,
        disk_space: float | None = None,
        onstart_cmd: str | None = None,
        env_vars: dict[str, str] | None = None,
        jupyter: bool = False,
        ssh: bool = True,
    ) -> dict[str, Any]:
        """
        Launch a new GPU instance.

        Args:
            gpu_name: GPU model name (e.g., "RTX_3090", "RTX_4090").
                Defaults to config value if not specified.
            image: Docker image to use. Can be a full image name or
                a key from DOCKER_IMAGES. Defaults to config value.
            num_gpus: Number of GPUs to rent.
                Defaults to config value if not specified.
            disk_space: Disk space in GB. Defaults to config value.
            onstart_cmd: Command to run on instance start.
            env_vars: Environment variables as key-value pairs.
            jupyter: Enable Jupyter notebook.
            ssh: Enable SSH access.

        Returns:
            Instance creation response from API.
        """
        # Use config defaults if not specified
        gpu_name = gpu_name if gpu_name is not None else self.config.gpu_name
        image = image if image is not None else self.config.docker_image
        num_gpus = num_gpus if num_gpus is not None else self.config.num_gpus
        disk_space = disk_space if disk_space is not None else self.config.disk_space

        # Resolve image shortcut
        image = DOCKER_IMAGES.get(image, image)

        kwargs: dict[str, Any] = {
            "num_gpus": str(num_gpus),
            "gpu_name": gpu_name,
            "image": image,
            "disk": str(disk_space),
        }

        if onstart_cmd:
            kwargs["onstart_cmd"] = onstart_cmd

        if env_vars:
            env_str = " ".join(f"-e {k}={v}" for k, v in env_vars.items())
            kwargs["env"] = env_str

        if jupyter:
            kwargs["jupyter"] = True
            kwargs["jupyter_dir"] = "/workspace"

        if ssh:
            kwargs["ssh"] = True

        logger.info("Launching instance: gpu=%s, image=%s", gpu_name, image)
        return self.sdk.launch_instance(**kwargs)

    def launch_by_offer_id(
        self,
        offer_id: int,
        image: str | None = None,
        disk_space: float | None = None,
        onstart_cmd: str | None = None,
        env_vars: dict[str, str] | None = None,
        jupyter: bool = False,
        ssh: bool = True,
    ) -> dict[str, Any]:
        """
        Launch a new GPU instance by offer ID.

        Args:
            offer_id: The specific offer ID from search results.
            image: Docker image to use.
            disk_space: Disk space in GB.
            onstart_cmd: Command to run on instance start.
            env_vars: Environment variables as key-value pairs.
            jupyter: Enable Jupyter notebook.
            ssh: Enable SSH access.

        Returns:
            Instance creation response from API.
        """
        image = image if image is not None else self.config.docker_image
        disk_space = disk_space if disk_space is not None else self.config.disk_space

        # Resolve image shortcut
        image = DOCKER_IMAGES.get(image, image)

        kwargs: dict[str, Any] = {
            "id": offer_id,
            "image": image,
            "disk": str(disk_space),
        }

        if onstart_cmd:
            kwargs["onstart_cmd"] = onstart_cmd

        if env_vars:
            env_str = " ".join(f"-e {k}={v}" for k, v in env_vars.items())
            kwargs["env"] = env_str

        if jupyter:
            kwargs["jupyter"] = True
            kwargs["jupyter_dir"] = "/workspace"

        if ssh:
            kwargs["ssh"] = True

        logger.info("Launching instance from offer %d with image=%s", offer_id, image)
        return self.sdk.create_instance(**kwargs)

    def list_instances(self) -> list[dict[str, Any]]:
        """
        List all your instances.

        Returns:
            List of instance dictionaries from API.
        """
        result = self.sdk.show_instances()
        logger.debug("Found %d instances", len(result) if isinstance(result, list) else 0)
        return result

    def get_instance(self, instance_id: int) -> dict[str, Any] | None:
        """
        Get details of a specific instance.

        Args:
            instance_id: The instance ID.

        Returns:
            Instance details dictionary or None if not found.
        """
        instances = self.list_instances()
        if isinstance(instances, list):
            for inst in instances:
                if inst.get("id") == instance_id:
                    return inst
        return None

    def start_instance(self, instance_id: int) -> dict[str, Any]:
        """
        Start a stopped instance.

        Args:
            instance_id: The instance ID to start.

        Returns:
            API response dictionary.
        """
        logger.info("Starting instance %d", instance_id)
        return self.sdk.start_instance(id=instance_id)

    def stop_instance(self, instance_id: int) -> dict[str, Any]:
        """
        Stop a running instance.

        Stops the instance but keeps it available. This stops billing
        but allows you to restart it later.

        Args:
            instance_id: The instance ID to stop.

        Returns:
            API response dictionary.
        """
        logger.info("Stopping instance %d", instance_id)
        return self.sdk.stop_instance(id=instance_id)

    def destroy_instance(self, instance_id: int) -> dict[str, Any]:
        """
        Destroy an instance completely.

        This permanently removes the instance and all data on it.

        Args:
            instance_id: The instance ID to destroy.

        Returns:
            API response dictionary.
        """
        logger.warning("Destroying instance %d", instance_id)
        return self.sdk.destroy_instance(id=instance_id)

    def get_balance(self) -> dict[str, Any]:
        """
        Get account balance and user info.

        Returns:
            Dictionary with credit, total_spend, and other user info.
        """
        user = self.sdk.show_user()
        return {
            "credit": user.get("credit", 0),
            "total_spend": abs(user.get("total_spend", 0)),
            "username": user.get("username"),
            "email": user.get("email"),
        }

    def get_invoices(self, limit: int = 20) -> list[dict[str, Any]]:
        """
        Get recent invoices/transactions.

        Args:
            limit: Maximum number of invoices to return.

        Returns:
            List of invoice dictionaries.
        """
        invoices = self.sdk.show_invoices()
        if isinstance(invoices, list):
            return invoices[:limit]
        return []

    def add_ssh_key(self, public_key: str) -> dict[str, Any]:
        """
        Add an SSH public key to your account.

        Args:
            public_key: SSH public key string.

        Returns:
            API response dictionary.
        """
        logger.info("Adding SSH key")
        return self.sdk.create_ssh_key(ssh_key=public_key)

    def list_ssh_keys(self) -> list[dict[str, Any]]:
        """
        List all SSH keys on your account.

        Returns:
            List of SSH key dictionaries.
        """
        return self.sdk.show_ssh_keys()

    def get_ssh_command(self, instance_id: int) -> str | None:
        """
        Get the SSH command to connect to an instance.

        Args:
            instance_id: The instance ID.

        Returns:
            SSH command string or None if instance not found or
            SSH is not available.
        """
        instance = self.get_instance(instance_id)
        if instance:
            host = instance.get("ssh_host")
            port = instance.get("ssh_port")
            if host and port:
                return f"ssh -p {port} root@{host}"
        return None


def quick_launch(
    gpu_name: str | None = None,
    image: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Quick helper to launch a single GPU instance.

    This is a convenience function for quickly launching an instance
    with minimal configuration.

    Args:
        gpu_name: GPU model name. Uses default if not specified.
        image: Docker image. Uses default if not specified.
        api_key: Vast.ai API key. Uses environment if not specified.

    Returns:
        Instance creation response from API.

    Example:
        >>> result = quick_launch(gpu_name="RTX_4090")
    """
    manager = VastGPUManager(api_key=api_key)
    return manager.launch_instance(gpu_name=gpu_name, image=image)


if __name__ == "__main__":
    import json

    # Configure logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        manager = VastGPUManager()

        print("Searching for available GPUs...")
        offers = manager.search_gpus(limit=5)

        if isinstance(offers, list) and offers:
            print(f"\nFound {len(offers)} offers:")
            print(json.dumps(offers[0], indent=2, default=str))
        else:
            print("No offers found matching criteria")

        print("\nYour current instances:")
        instances = manager.list_instances()
        print(json.dumps(instances, indent=2, default=str))

    except APIKeyError as e:
        print(f"Error: {e}")
        print("\nSet your API key in .env file or environment:")
        print("  VASTAI_API_KEY='your_api_key_here'")
