
"""
app/services/memory.py

Custom MongoDB checkpoint saver for LangGraph
"""

from typing import Any, Optional, Iterator, Sequence
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from pymongo.collection import Collection
import pickle
from datetime import datetime

class MongoSaver(BaseCheckpointSaver):
    """
    Custom checkpoint saver that stores LangGraph state in MongoDB.
    """
    
    def __init__(self, collection: Collection):
        """
        Initialize MongoSaver with a MongoDB collection.
        
        Args:
            collection: PyMongo collection object for storing checkpoints
        """
        super().__init__()
        self.collection = collection
        
        # Create indexes for better query performance
        self.collection.create_index([("thread_id", 1), ("checkpoint_ns", 1)])
        self.collection.create_index([("thread_id", 1), ("checkpoint_id", 1)])
    
    def get_next_version(self, current: Optional[int], channel: Any) -> int:
        """
        Generate the next version number for a checkpoint.
        
        Args:
            current: Current version number (or None for first version)
            channel: The channel being versioned
            
        Returns:
            Next version number
        """
        if current is None:
            return 1
        return current + 1
    
    def put(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, int],
    ) -> dict[str, Any]:
        """
        Save a checkpoint to MongoDB.
        
        Args:
            config: Configuration dict with thread_id and checkpoint_ns
            checkpoint: The checkpoint data to save
            metadata: Metadata associated with the checkpoint
            new_versions: Dictionary of channel versions
            
        Returns:
            Updated config dict
        """
        thread_id = config["configurable"]["thread_id"]
        user_id = config["configurable"].get("user_id")
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        
        # Serialize checkpoint and metadata using pickle
        checkpoint_data = pickle.dumps(checkpoint)
        metadata_data = pickle.dumps(metadata)
        
        # Create document to store
        document = {
            "doc_type": "checkpoint",  # ✅ Add this
            "thread_id": thread_id,
            "user_id": user_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "checkpoint": checkpoint_data,
            "metadata": metadata_data,
            "created_at": datetime.utcnow(),
        }
        
        # Upsert (update if exists, insert if not)
        self.collection.update_one(
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            },
            {"$set": document},
            upsert=True,
        )
        
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }
    
    def put_writes(
        self,
        config: dict[str, Any],
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """
        Store intermediate writes during graph execution.
        
        Args:
            config: Configuration dict with thread_id and checkpoint_ns
            writes: Sequence of (channel, value) tuples to write
            task_id: Unique identifier for this task
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")
        
        # Serialize writes
        writes_data = pickle.dumps(writes)
        
        # Create document to store writes
        document = {
            "doc_type": "writes",  # ✅ Add this
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "task_id": task_id,
            "writes": writes_data,
            "created_at": datetime.utcnow(),
        }
        
        # Insert the writes document
        # Use a different collection or add a type field to distinguish from checkpoints
        self.collection.insert_one(document)
    
    def get_tuple(self, config: dict[str, Any]) -> Optional[CheckpointTuple]:
        """
        Retrieve a checkpoint from MongoDB.

        Args:
            config: Configuration dict with thread_id and checkpoint_ns

        Returns:
            CheckpointTuple if found, None otherwise
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        # Build query
        query = {
            "doc_type": "checkpoint",  # ✅ Add this filter
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }

        if checkpoint_id:
            query["checkpoint_id"] = checkpoint_id

        # Find the most recent checkpoint
        document = self.collection.find_one(
            query,
            sort=[("created_at", -1)]
        )

        if not document:
            return None

        # ✅ Safely handle missing fields
        if "checkpoint" not in document:
            print(f"⚠️ Missing 'checkpoint' in document with _id={document.get('_id')}")
            return None

        if "metadata" not in document:
            print(f"⚠️ Missing 'metadata' in document with _id={document.get('_id')}")
            return None

        # Deserialize safely
        try:
            checkpoint = pickle.loads(document["checkpoint"])
        except Exception as e:
            print(f"❌ Failed to unpickle 'checkpoint' for document {document.get('_id')}: {e}")
            return None

        try:
            metadata = pickle.loads(document["metadata"])
        except Exception as e:
            print(f"❌ Failed to unpickle 'metadata' for document {document.get('_id')}: {e}")
            metadata = {}

        # Build checkpoint config
        checkpoint_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": document.get("checkpoint_id"),
            }
        }

        # Handle parent checkpoint
        parent_checkpoint_id = checkpoint.get("parent_checkpoint_id") if checkpoint else None
        parent_config = None
        if parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_checkpoint_id,
                }
            }

        return CheckpointTuple(
            config=checkpoint_config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
        )

    
    def list(
        self,
        config: Optional[dict[str, Any]],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """
        List checkpoints from MongoDB.
        
        Args:
            config: Optional configuration dict with thread_id
            filter: Optional filter criteria
            before: Optional checkpoint to start before
            limit: Optional maximum number of checkpoints to return
            
        Yields:
            CheckpointTuple objects
        """
        query = {
            "doc_type": "checkpoint"  # ✅ Add this filter
        }
        
        if config:
            thread_id = config.get("configurable", {}).get("thread_id")
            if thread_id:
                query["thread_id"] = thread_id
            
            checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
            query["checkpoint_ns"] = checkpoint_ns
        
        if filter:
            query.update(filter)
        
        # Apply before filter if provided
        if before:
            before_id = before.get("configurable", {}).get("checkpoint_id")
            if before_id:
                # Find the checkpoint to get its timestamp
                before_doc = self.collection.find_one({"checkpoint_id": before_id})
                if before_doc:
                    query["created_at"] = {"$lt": before_doc["created_at"]}
        
        # Query with sorting and limit
        cursor = self.collection.find(
            query,
            sort=[("created_at", -1)]
        )
        
        if limit:
            cursor = cursor.limit(limit)
        
        for document in cursor:
            # Skip documents without checkpoint data (e.g., writes)
            if "checkpoint" not in document:
                continue
                
            checkpoint = pickle.loads(document["checkpoint"])
            metadata = pickle.loads(document["metadata"])
            
            checkpoint_config = {
                "configurable": {
                    "thread_id": document["thread_id"],
                    "checkpoint_ns": document["checkpoint_ns"],
                    "checkpoint_id": document["checkpoint_id"],
                }
            }
            
            parent_checkpoint_id = checkpoint.get("parent_checkpoint_id")
            parent_config = None
            if parent_checkpoint_id:
                parent_config = {
                    "configurable": {
                        "thread_id": document["thread_id"],
                        "checkpoint_ns": document["checkpoint_ns"],
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
            
            yield CheckpointTuple(
                config=checkpoint_config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
            )
