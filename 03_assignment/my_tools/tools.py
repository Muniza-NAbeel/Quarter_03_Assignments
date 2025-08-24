from agents import function_tool, RunContextWrapper

ORDER_DB = {"123": "Shipped", "456": "Processing", "789": "Delivered"}

def is_order_query(ctx: RunContextWrapper, agent) -> bool:
    text = str(getattr(ctx, "context", {}) .get("user_input", "")).lower()
    return any(k in text for k in ["order", "status"])

@function_tool
def get_order_status(order_id: str) -> str:
    """Fetches the order status from a fake database."""
    status = ORDER_DB.get(order_id)
    if not status:
        raise ValueError("Order ID not found")
    return f"Order {order_id} is currently '{status}'."