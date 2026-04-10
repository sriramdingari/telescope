def run():
    config = load_config()
    client = redis.from_url("redis://localhost")
    return client.get(config.key)
