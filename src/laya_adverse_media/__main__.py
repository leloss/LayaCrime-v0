import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env.local")
    uvicorn.run(
        "laya_adverse_media.app:app",
        host=os.getenv("LAYA_HOST", "127.0.0.1"),
        port=int(os.getenv("LAYA_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
