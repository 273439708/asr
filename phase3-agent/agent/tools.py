"""模拟智能座舱工具 + 车辆状态（模拟 ECU）。

每个工具用 @tool 装饰，LangChain 自动从类型注解 + docstring 抽出 JSON Schema 供
LLM function calling。工具只读/改模块级 VEHICLE 字典（模拟车机内存），返回给用户的
自然语言确认句由工具自己组织——这样 LLM 拿到工具结果后能直接复述执行情况。

座舱三域（车控/导航/媒体）：
  - 车控 climate/window/seat_heat
  - 导航 navigate/find_poi
  - 媒体 media_play/media_volume/media_pause
"""

from langchain_core.tools import tool

# 模拟车辆当前状态（进程内，相当于车机 ECU 的一份镜像）
VEHICLE = {
    "climate_temp": 24.0,       # 空调设定温度 ℃
    "climate_on": True,
    "windows": {"主驾": 0, "副驾": 0, "左后": 0, "右后": 0},  # 开度 0-100%
    "seat_heat": {"主驾": 0, "副驾": 0},                      # 加热档 0-3
    "nav_destination": None,
    "media_playing": None,
    "media_volume": 30,         # 0-100
}

WINDOW_NAMES = {"主驾", "副驾", "左后", "右后", "全部"}
SEAT_NAMES = {"主驾", "副驾"}


@tool
def set_climate_temperature(temperature: float) -> str:
    """设置空调温度。temperature 为目标摄氏温度，常见范围 16-32 度。"""
    temperature = max(16.0, min(32.0, temperature))
    VEHICLE["climate_temp"] = temperature
    VEHICLE["climate_on"] = True
    return f"空调温度已设为 {temperature:.0f} 度"


@tool
def control_window(position: str, open_percent: int) -> str:
    """控制车窗开合。position 取值：主驾/副驾/左后/右后/全部。
    open_percent 为开度百分比，0 表示完全关闭，100 表示完全打开。"""
    if position not in WINDOW_NAMES:
        return f"无法识别车窗位置「{position}」，可选：主驾/副驾/左后/右后/全部"
    open_percent = max(0, min(100, open_percent))
    targets = list(VEHICLE["windows"]) if position == "全部" else [position]
    for t in targets:
        VEHICLE["windows"][t] = open_percent
    where = "所有车窗" if position == "全部" else f"{position}车窗"
    action = "关闭" if open_percent == 0 else f"打开到 {open_percent}%"
    return f"已将{where}{action}"


@tool
def set_seat_heating(seat: str, level: int) -> str:
    """设置座椅加热档位。seat 取值：主驾/副驾。level 为加热档位 0-3，0 表示关闭。"""
    if seat not in SEAT_NAMES:
        return f"无法识别座椅「{seat}」，可选：主驾/副驾"
    level = max(0, min(3, level))
    VEHICLE["seat_heat"][seat] = level
    if level == 0:
        return f"已关闭{seat}座椅加热"
    return f"已开启{seat}座椅加热 {level} 档"


@tool
def navigate_to(destination: str) -> str:
    """设置导航目的地并开始导航。destination 为目的地名称或地址。"""
    VEHICLE["nav_destination"] = destination
    return f"已开始导航到「{destination}」"


@tool
def find_nearby(category: str) -> str:
    """查找附近的兴趣点（POI）。category 为类别，如：加油站/充电站/餐厅/停车场/超市。"""
    # 模拟检索结果（端侧 demo 不接真实地图 API）
    fake = {
        "加油站": ["中石化(人民路站) 1.2km", "中石油(解放路站) 2.0km"],
        "充电站": ["特来电快充站 0.8km", "国家电网充电站 1.5km"],
        "餐厅": ["海底捞(万达店) 0.5km", "肯德基(中心广场店) 0.9km"],
        "停车场": ["万达广场地下停车场 0.3km", "市民中心停车场 1.1km"],
        "超市": ["永辉超市 0.6km", "沃尔玛 1.8km"],
    }
    hits = fake.get(category, [f"{category}(模拟结果A) 1.0km", f"{category}(模拟结果B) 2.3km"])
    return f"为您找到附近的{category}：" + "；".join(hits)


@tool
def media_play(query: str) -> str:
    """播放媒体（音乐/电台/有声书）。query 为歌曲名、歌手或电台名称。"""
    VEHICLE["media_playing"] = query
    return f"正在为您播放「{query}」"


@tool
def set_media_volume(volume: int) -> str:
    """设置媒体音量。volume 为目标音量 0-100。"""
    volume = max(0, min(100, volume))
    VEHICLE["media_volume"] = volume
    return f"音量已调节到 {volume}"


@tool
def media_pause() -> str:
    """暂停当前媒体播放。"""
    cur = VEHICLE["media_playing"]
    VEHICLE["media_playing"] = None
    return "已暂停播放" + (f"「{cur}」" if cur else "")


ALL_TOOLS = [
    set_climate_temperature, control_window, set_seat_heating,
    navigate_to, find_nearby,
    media_play, set_media_volume, media_pause,
]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def reset_vehicle():
    """每条指令独立演示时重置车辆状态，便于观察单条指令的状态增量。"""
    VEHICLE.update({
        "climate_temp": 24.0, "climate_on": True,
        "windows": {"主驾": 0, "副驾": 0, "左后": 0, "右后": 0},
        "seat_heat": {"主驾": 0, "副驾": 0},
        "nav_destination": None, "media_playing": None, "media_volume": 30,
    })


def vehicle_snapshot() -> dict:
    """返回当前车辆状态的浅拷贝快照（用于打印对比）。"""
    import copy
    return copy.deepcopy(VEHICLE)
