from aiohttp import web


async def health(_):
    return web.json_response({"status": "ok"})


async def build_app():
    app = web.Application()
    app.router.add_get("/healthz", health)
    return app


if __name__ == "__main__":
    web.run_app(build_app(), port=8080)