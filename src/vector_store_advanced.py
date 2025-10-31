import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
import json


class AdvancedVectorStore:
    """
    Production-ready vector store with persistent storage
    """
    
    def __init__(
        self,
        collection_name: str = "data_assistant",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize advanced vector store with persistence
        
        Args:
            collection_name: Name for the collection
            persist_directory: Directory to persist data
            embedding_model: Sentence transformer model to use
        """
        # Create persist directory if it doesn't exist
        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding function
        self.embedding_model_name = embedding_model
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Create or get collection with embedding function
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "description": "Advanced data analysis assistant with RAG",
                "embedding_model": embedding_model
            }
        )
        
        print(f"✅ Advanced vector store initialized")
        print(f"   📁 Persist: {self.persist_dir}")
        print(f"   🧠 Model: {embedding_model}")
        print(f"   📊 Documents: {self.collection.count()}")
    
    def add_dataframe_context(
        self,
        df: pd.DataFrame,
        file_name: str,
        chunk_size: int = 50
    ) -> int:
        """
        Add dataframe to vector store with smart chunking
        
        Args:
            df: Pandas DataFrame
            file_name: Name of the dataset
            chunk_size: Number of rows per chunk
            
        Returns:
            Number of documents added
        """
        documents = []
        metadatas = []
        ids = []
        doc_count = 0
        
        # 1. Add column metadata (one document per column)
        for idx, col in enumerate(df.columns):
            col_doc = self._create_column_document(df, col)
            documents.append(col_doc)
            metadatas.append({
                "type": "column_metadata",
                "column_name": col,
                "file_name": file_name,
                "data_type": str(df[col].dtype)
            })
            ids.append(f"{file_name}_col_{idx}")
            doc_count += 1
        
        # 2. Add data chunks (multiple rows per document)
        num_chunks = (len(df) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(df))
            chunk_df = df.iloc[start_idx:end_idx]
            
            chunk_doc = self._create_chunk_document(chunk_df, start_idx, end_idx)
            documents.append(chunk_doc)
            metadatas.append({
                "type": "data_chunk",
                "file_name": file_name,
                "start_row": start_idx,
                "end_row": end_idx,
                "num_rows": end_idx - start_idx
            })
            ids.append(f"{file_name}_chunk_{chunk_idx}")
            doc_count += 1
        
        # 3. Add statistical summary
        stats_doc = self._create_stats_document(df)
        documents.append(stats_doc)
        metadatas.append({
            "type": "statistics",
            "file_name": file_name
        })
        ids.append(f"{file_name}_stats")
        doc_count += 1
        
        # 4. Add categorical summaries
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for idx, col in enumerate(categorical_cols):
            if df[col].nunique() <= 50:  # Only if reasonable number of categories
                cat_doc = self._create_categorical_document(df, col)
                documents.append(cat_doc)
                metadatas.append({
                    "type": "categorical_summary",
                    "column_name": col,
                    "file_name": file_name
                })
                ids.append(f"{file_name}_cat_{idx}")
                doc_count += 1
        
        # Add all documents to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Added {doc_count} documents from {file_name}")
        return doc_count
    
    def semantic_search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Perform semantic search with optional filtering
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            Search results
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_metadata
        )
        
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0]
        }
    
    def hybrid_search(
        self,
        query: str,
        keywords: Optional[List[str]] = None,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Hybrid search combining semantic + keyword matching
        
        Args:
            query: Semantic search query
            keywords: Optional keywords to boost
            n_results: Number of results
            
        Returns:
            Combined search results
        """
        # Perform semantic search
        semantic_results = self.semantic_search(query, n_results=n_results * 2)
        
        # If keywords provided, boost matching results
        if keywords:
            scored_results = []
            for doc, meta, dist in zip(
                semantic_results["documents"],
                semantic_results["metadatas"],
                semantic_results["distances"]
            ):
                score = 1.0 - dist  # Convert distance to similarity
                
                # Boost score if keywords found
                doc_lower = doc.lower()
                keyword_matches = sum(1 for kw in keywords if kw.lower() in doc_lower)
                score += keyword_matches * 0.2
                
                scored_results.append({
                    "document": doc,
                    "metadata": meta,
                    "score": score
                })
            
            # Sort by score and take top n
            scored_results.sort(key=lambda x: x["score"], reverse=True)
            top_results = scored_results[:n_results]
            
            return {
                "documents": [r["document"] for r in top_results],
                "metadatas": [r["metadata"] for r in top_results],
                "scores": [r["score"] for r in top_results]
            }
        
        # Return semantic results if no keywords
        return {
            "documents": semantic_results["documents"][:n_results],
            "metadatas": semantic_results["metadatas"][:n_results],
            "scores": [1.0 - d for d in semantic_results["distances"][:n_results]]
        }
    
    def get_enhanced_context(
        self,
        question: str,
        n_results: int = 3,
        include_keywords: bool = True
    ) -> str:
        """
        Get enhanced context for RAG with automatic keyword extraction
        
        Args:
            question: User's question
            n_results: Number of context pieces
            include_keywords: Whether to use hybrid search
            
        Returns:
            Formatted context string
        """
        # Extract potential keywords from question
        keywords = None
        if include_keywords:
            keywords = self._extract_keywords(question)
        
        # Perform hybrid search
        if keywords:
            results = self.hybrid_search(question, keywords, n_results)
        else:
            results = self.semantic_search(question, n_results)
        
        # Format context
        context_parts = []
        for doc, meta, score in zip(
            results["documents"],
            results["metadatas"],
            results.get("scores", [1.0] * len(results["documents"]))
        ):
            context_parts.append(
                f"[{meta['type']} | {meta['file_name']} | relevance: {score:.2f}]\n{doc}"
            )
        
        return "\n\n" + "="*80 + "\n\n".join(context_parts)
    
    def _create_column_document(self, df: pd.DataFrame, column: str) -> str:
        """Create rich document for column metadata"""
        col_data = df[column]
        
        parts = [
            f"Column Name: {column}",
            f"Data Type: {col_data.dtype}",
            f"Non-null Count: {col_data.count()} out of {len(df)}",
            f"Null Count: {col_data.isnull().sum()}"
        ]
        
        if col_data.dtype in ['int64', 'float64']:
            parts.extend([
                f"Mean: {col_data.mean():.2f}",
                f"Median: {col_data.median():.2f}",
                f"Min: {col_data.min():.2f}",
                f"Max: {col_data.max():.2f}",
                f"Std Dev: {col_data.std():.2f}"
            ])
        else:
            unique_count = col_data.nunique()
            parts.append(f"Unique Values: {unique_count}")
            if unique_count <= 10:
                parts.append(f"Values: {', '.join(map(str, col_data.unique()))}")
        
        return " | ".join(parts)
    
    def _create_chunk_document(
        self,
        chunk_df: pd.DataFrame,
        start_idx: int,
        end_idx: int
    ) -> str:
        """Create document for data chunk"""
        parts = [f"Data rows {start_idx} to {end_idx}:"]
        
        # Convert each row to readable text
        for idx, row in chunk_df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
            parts.append(f"Row {idx}: {row_text}")
        
        return "\n".join(parts)
    
    def _create_stats_document(self, df: pd.DataFrame) -> str:
        """Create statistical summary document"""
        parts = [
            f"Dataset Overview:",
            f"Total Rows: {len(df)}",
            f"Total Columns: {len(df.columns)}",
            f"Column Names: {', '.join(df.columns)}"
        ]
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            parts.append(f"\nNumeric Columns Statistics:")
            for col in numeric_cols:
                parts.append(
                    f"{col}: mean={df[col].mean():.2f}, "
                    f"min={df[col].min():.2f}, "
                    f"max={df[col].max():.2f}"
                )
        
        return "\n".join(parts)
    
    def _create_categorical_document(self, df: pd.DataFrame, column: str) -> str:
        """Create document for categorical column summary"""
        value_counts = df[column].value_counts()
        
        parts = [f"Category Analysis for '{column}':"]
        parts.append(f"Total Unique Values: {len(value_counts)}")
        parts.append("\nTop Values:")
        
        for val, count in value_counts.head(20).items():
            percentage = (count / len(df)) * 100
            parts.append(f"  - {val}: {count} occurrences ({percentage:.1f}%)")
        
        return "\n".join(parts)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract potential keywords from text"""
        # Remove common words
        common_words = {
            'what', 'which', 'how', 'show', 'tell', 'get', 'find', 
            'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'has', 'have', 'had', 'do', 'does', 'did', 'can', 'could',
            'should', 'would', 'me', 'my', 'about', 'from'
        }
        
        words = text.lower().split()
        keywords = [w.strip('?.,!') for w in words if w not in common_words and len(w) > 2]
        
        return keywords[:5]  # Return top 5 keywords
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name,
            "embedding_model": self.embedding_model_name,
            "persist_directory": str(self.persist_dir)
        }
    
    def reset(self) -> None:
        """Clear all data from vector store"""
        self.client.reset()
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_function
        )
        print("✅ Vector store reset")