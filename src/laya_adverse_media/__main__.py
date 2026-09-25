import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "laya_adverse_media.app:app",
        host=os.getenv("LAYA_HOST", "127.0.0.1"),
        port=int(os.getenv("LAYA_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
