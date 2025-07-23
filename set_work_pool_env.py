import asyncio
from dotenv import dotenv_values
from prefect.client.orchestration import get_client
from prefect.client.schemas.actions import WorkPoolUpdate


async def set_env_on_work_pool():
    pool_name = "ml-pool"
    env_vars = dotenv_values(".env")

    async with get_client() as client:
        # Fetch the current job template
        pool = await client.read_work_pool(pool_name)
        base_template = pool.base_job_template

        # Merge in new env vars
        if "variables" not in base_template:
            base_template["variables"] = {}

        if "env" not in base_template["variables"]:
            base_template["variables"]["env"] = {}

        base_template["variables"]["env"].update(env_vars)

        update = WorkPoolUpdate(base_job_template=base_template)

        await client.update_work_pool(pool_name, update)

        print(f"Environment variables successfully updated on work pool: {pool_name}")

asyncio.run(set_env_on_work_pool())
