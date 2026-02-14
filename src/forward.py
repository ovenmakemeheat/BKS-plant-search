import os
import time

import ngrok
from dotenv import load_dotenv


def start_ngrok_tunnel(port: int = 8000):
    """
    Start an ngrok tunnel for the given port.

    Args:
        port: The port to forward (default: 8000)

    Returns:
        The ngrok listener URL
    """
    # docs: https://ngrok.github.io/ngrok-python/
    load_dotenv()

    listener = ngrok.forward(
        # The port your app is running on.
        port,
        authtoken=os.getenv("NGROK_AUTHTOKEN"),
        domain=os.getenv("NGROK_DOMAIN"),
        # Secure your endpoint with a Traffic Policy.
        # This could also be a path to a Traffic Policy file.
        # traffic_policy='{"on_http_request": [{"actions": [{"type": "oauth","config": {"provider": "google"}}]}]}',
    )

    # Output ngrok URL to console
    print(f"Ingress established at {listener.url()}")

    return listener.url()


# Keep the listener alive when run directly
if __name__ == "__main__":
    try:
        start_ngrok_tunnel(8000)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Closing listener")
