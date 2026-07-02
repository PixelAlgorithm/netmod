from netmiko import ConnectHandler

from settings import DeviceSettings, get_device_settings


def build_device_params(settings: DeviceSettings | None = None, **overrides) -> dict:
    settings = settings or get_device_settings()
    params = {
        "device_type": settings.device_type,
        "host": settings.host,
        "port": settings.port,
        "username": settings.username,
        "password": settings.password,
        "timeout": 120,
        "session_timeout": 120,
        "conn_timeout": 60,
        "banner_timeout": 60,
        "auth_timeout": 60,
        "read_timeout_override": 120,
        "fast_cli": False,
        "global_delay_factor": 2,
    }
    params.update(overrides)
    return params


def open_device_connection(settings: DeviceSettings | None = None, **overrides):
    return ConnectHandler(**build_device_params(settings, **overrides))
