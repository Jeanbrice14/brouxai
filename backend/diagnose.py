"""Script de diagnostic du pipeline BrouxAI.

Lance via : python diagnose.py  (dans le dossier backend, avec le venv activé)
"""
from __future__ import annotations

import asyncio
import io
import sys


async def check_redis():
    print("\n=== Redis ===")
    try:
        import redis.asyncio as aioredis
        client = aioredis.from_url("redis://localhost:6379/0", decode_responses=True)
        pong = await client.ping()
        print(f"  ✓ Redis OK — ping={pong}")
        await client.aclose()
        return True
    except Exception as e:
        print(f"  ✗ Redis KO : {e}")
        return False


async def check_minio():
    print("\n=== MinIO / Storage ===")
    try:
        import asyncio
        import boto3

        def _ping():
            client = boto3.client(
                "s3",
                endpoint_url="http://localhost:9000",
                aws_access_key_id="minioadmin",
                aws_secret_access_key="minioadmin",
                region_name="us-east-1",
            )
            buckets = client.list_buckets()
            names = [b["Name"] for b in buckets.get("Buckets", [])]
            return names

        buckets = await asyncio.to_thread(_ping)
        print(f"  ✓ MinIO OK — buckets: {buckets}")

        if "narr8-dev" not in buckets:
            print("  ! Bucket 'narr8-dev' n'existe pas → création...")
            def _create():
                import boto3
                client = boto3.client(
                    "s3",
                    endpoint_url="http://localhost:9000",
                    aws_access_key_id="minioadmin",
                    aws_secret_access_key="minioadmin",
                    region_name="us-east-1",
                )
                client.create_bucket(Bucket="narr8-dev")
                print("  ✓ Bucket 'narr8-dev' créé !")
            await asyncio.to_thread(_create)
        else:
            print("  ✓ Bucket 'narr8-dev' existe")

        return True
    except Exception as e:
        print(f"  ✗ MinIO KO : {e}")
        return False


async def check_llm():
    print("\n=== LLM (OpenAI) ===")
    try:
        import litellm
        litellm.suppress_debug_info = True
        response = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": 'Retourne {"ok": true}'}],
            temperature=0.0,
        )
        content = response.choices[0].message.content
        print(f"  ✓ LLM OK — réponse: {content[:80]}")
        return True
    except Exception as e:
        print(f"  ✗ LLM KO : {e}")
        return False


async def check_upload_and_metadata():
    print("\n=== Upload + MetadataAgent ===")
    try:
        import pandas as pd
        import sys
        import os

        # Ajouter le dossier courant au path
        sys.path.insert(0, os.path.dirname(__file__))

        # Charger les settings
        os.chdir(os.path.dirname(__file__) or ".")

        from app.services.storage import upload_file, read_dataframe

        # Créer un CSV de test
        csv_content = b"region,ca,segment\nNord,150000,B2B\nSud,120000,B2C\nEst,90000,B2B\n"
        ref = "s3://narr8-dev/demo-tenant/datasets/diagnose-test/ventes_test.csv"

        await upload_file(ref, csv_content, "text/csv")
        print("  ✓ Upload fichier test OK")

        df = await read_dataframe(ref)
        print(f"  ✓ Lecture DataFrame OK — {len(df)} lignes, colonnes: {list(df.columns)}")

        # Tester MetadataAgent
        from app.agents.metadata_agent import MetadataAgent
        from app.pipeline.state import initial_state

        agent = MetadataAgent()
        state = initial_state(
            tenant_id="demo-tenant",
            user_id="demo-user",
            report_id="diagnose-test-001",
            prompt="Analyse les ventes par région",
            raw_data_refs=[ref],
            brand_kit={},
        )
        state["status"] = "running"

        print("  → Lancement MetadataAgent...")
        result = await agent(state)

        if result.get("status") == "error":
            print(f"  ✗ MetadataAgent ERREUR : {result.get('errors')}")
            return False
        else:
            print(f"  ✓ MetadataAgent OK — status={result['status']}, hitl={result.get('hitl_pending')}")
            if result.get("metadata"):
                files = result["metadata"].get("files", {})
                for fref, fmeta in files.items():
                    print(f"    - {fref}: {fmeta['row_count']} lignes, avg_confidence={fmeta['avg_confidence']}")
            return True

    except Exception as e:
        import traceback
        print(f"  ✗ ERREUR : {e}")
        traceback.print_exc()
        return False


async def check_full_pipeline():
    print("\n=== Pipeline complet (test rapide) ===")
    try:
        import os, sys
        sys.path.insert(0, os.path.dirname(__file__) or ".")
        os.chdir(os.path.dirname(__file__) or ".")

        from app.pipeline.graph import build_pipeline
        from app.pipeline.state import initial_state
        from app.services.storage import upload_file

        # Upload CSV test
        csv_content = b"region,ca,segment\nNord,150000,B2B\nSud,120000,B2C\nEst,90000,B2B\n"
        ref = "s3://narr8-dev/demo-tenant/datasets/full-test/ventes_test.csv"
        await upload_file(ref, csv_content, "text/csv")

        pipeline = build_pipeline()
        state = initial_state(
            tenant_id="demo-tenant",
            user_id="demo-user",
            report_id="full-test-001",
            prompt="Analyse les ventes par région et identifie le top performer",
            raw_data_refs=[ref],
            brand_kit={},
        )
        state["status"] = "running"

        print("  → Lancement pipeline complet... (peut prendre 30-60s)")
        result = await pipeline.ainvoke(state)

        status = result.get("status", "unknown")
        errors = result.get("errors", [])
        current_agent = result.get("current_agent", "")
        hitl = result.get("hitl_pending", False)
        report_urls = result.get("report_urls", {})

        print(f"  Status final : {status}")
        print(f"  Dernier agent : {current_agent}")
        print(f"  HITL pending : {hitl}")
        print(f"  Report URLs : {report_urls}")
        if errors:
            print(f"  ERREURS : {errors}")
        return status not in ("error",)

    except Exception as e:
        import traceback
        print(f"  ✗ ERREUR : {e}")
        traceback.print_exc()
        return False


async def main():
    print("=" * 60)
    print("BrouxAI — Diagnostic pipeline")
    print("=" * 60)

    import os, sys
    sys.path.insert(0, os.path.dirname(__file__) or ".")
    os.chdir(os.path.dirname(__file__) or ".")

    redis_ok = await check_redis()
    minio_ok = await check_minio()

    if not redis_ok:
        print("\n⛔ Redis est DOWN. Lancez : docker compose up -d redis")
        sys.exit(1)

    if not minio_ok:
        print("\n⛔ MinIO est DOWN. Lancez : docker compose up -d minio")
        sys.exit(1)

    llm_ok = await check_llm()
    if not llm_ok:
        print("\n⛔ LLM KO. Vérifiez OPENAI_API_KEY dans .env")
        # Continuer quand même pour voir les autres erreurs

    meta_ok = await check_upload_and_metadata()

    if meta_ok:
        print("\n  → Tests MetadataAgent OK. Lancement du pipeline complet ? (y/n)")
        choice = input().strip().lower()
        if choice == "y":
            await check_full_pipeline()
    else:
        print("\n⛔ MetadataAgent KO — voir erreur ci-dessus")

    print("\n" + "=" * 60)
    print("Diagnostic terminé.")


if __name__ == "__main__":
    asyncio.run(main())
