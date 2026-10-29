
from qdrant_client import QdrantClient, models

LOCAL_PATH = "Final_Data/qdrant_db"
SERVER_URL = "http://localhost:6333"

COLLECTIONS = [
    "data_plane",
    "multi_access_edge",
    "l2-l3-protocol-logic",
    "physical_mac_layer",
    "o_ran_kpi",
    "o_ran_management",
    "ran_architecture",
]


def migrate():
    local = QdrantClient(path=LOCAL_PATH)
    remote = QdrantClient(url=SERVER_URL)

    try:
        for name in COLLECTIONS:
            print(f"\nMigrating: {name}")

            if not local.collection_exists(name):
                print(f"Collection not found locally: {name}")
                continue

            info = local.get_collection(name)
            print(f"Local points: {info.points_count}")

            if remote.collection_exists(name):
               remote_count = remote.count(
                   collection_name=name,
                   exact=True
               ).count
           
               local_count = info.points_count
           
               if remote_count == local_count:
                   print(f"Collection already migrated: {name}")
                   continue
           
               if remote_count > local_count:
                   raise RuntimeError(
                       f"{name}: server has more points than local. "
                       "Stopping to avoid unintended changes."
                   )
           
               print(
                   f"Resuming migration: {name}, "
                   f"server has {remote_count}/{local_count} points"
               )
            else:
                remote.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(
                        size=1024,
                        distance=models.Distance.COSINE,
                    ),
                )

            

            offset = None
            migrated = 0

            while True:
                points, next_offset = local.scroll(
                    collection_name=name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )

                if not points:
                    break

                converted_points = []

                for p in points:
                    if p.vector is None:
                        raise ValueError(f"Point {p.id} has no vector in {name}")
                
                    converted_points.append(
                        models.PointStruct(
                            id=p.id,
                            vector=p.vector,
                            payload=p.payload or {},
                        )
                    )
                
                remote.upsert(
                    collection_name=name,
                    points=converted_points,
                    wait=True,
                )
                
                migrated += len(points)
                print(f"Copied {migrated} points")

                if next_offset is None:
                    break

                offset = next_offset

            remote_count = remote.count(
                collection_name=name,
                exact=True,
            ).count

            print(f"Migration complete: {name}")
            print(f"Verified server point count: {remote_count}")

            if info.points_count is not None and remote_count != info.points_count:
                print("WARNING: Point counts do not match!")

        print("\nMigration process finished.")

    finally:
        local.close()
        remote.close()


if __name__ == "__main__":
    migrate()
