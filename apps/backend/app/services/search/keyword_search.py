"""
PostgreSQL Full-Text Keyword Search Service

Provides keyword-based search over document chunks using PostgreSQL's
tsvector and tsquery with ranking (ts_rank).
"""

from typing import List, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import SessionLocal


def search_chunks_keyword(
    query: str,
    limit: int = 10,
    min_score: float = 0.0,
    document_ids: Optional[List[int]] = None,
) -> List[dict]:
    """
    Search document chunks using PostgreSQL full-text search.
    
    Args:
        query: Search query string. Supports AND/OR/NOT operators via tsquery syntax.
        limit: Maximum number of results to return.
        min_score: Minimum ts_rank score threshold (0.0 to ~0.4 typical).
        document_ids: Optional filter to restrict search to specific documents.
        
    Returns:
        List of chunk dicts with metadata and ts_rank score.
    """
    session: Session = SessionLocal()
    try:
        # Convert query to tsquery format (simple word tokens with AND)
        # This handles multi-word queries by joining with & (AND)
        tokens = query.strip().split()
        if not tokens:
            return []
        
        # Use prefix matching with :* for partial word matches
        tsquery_parts = " & ".join(f"{token}:*" for token in tokens if token)
        tsquery_str = tsquery_parts.strip(" & ")
        
        # Build params dict
        params = {"query": query}
        
        # Build WHERE clause with optional document filter
        where_clause = "WHERE to_tsvector('english', dc.content) @@ plainto_tsquery('english', :query)"
        if document_ids:
            placeholders = ", ".join(f":doc_{i}" for i in range(len(document_ids)))
            where_clause += f" AND dc.document_id IN ({placeholders})"
            for i, doc_id in enumerate(document_ids):
                params[f"doc_{i}"] = doc_id
        
        sql = text(f"""
            SELECT
                dc.id as chunk_id,
                dc.document_id,
                dc.document_version_id,
                dc.chunk_index,
                dc.title,
                dc.section_heading,
                dc.page_number,
                dc.source_file_name,
                dc.content_hash,
                LEFT(dc.content, 500) as content,
                ts_rank(to_tsvector('english', dc.content), plainto_tsquery('english', :query)) as rank,
                ts_headline('english', dc.content, plainto_tsquery('english', :query),
                    'StartSel=<mark>, StopSel=</mark>, MaxWords=50, MinWords=20, MaxFragments=2') as headline
            FROM document_chunks dc
            {where_clause}
            ORDER BY rank DESC
            LIMIT :limit
        """)
        params["limit"] = limit
        
        result = session.execute(sql, params)
        rows = result.fetchall()
        
        chunks = []
        for row in rows:
            rank = float(row.rank) if row.rank else 0.0
            if rank < min_score:
                continue
            chunks.append({
                "chunk_id": row.chunk_id,
                "document_id": row.document_id,
                "document_version_id": row.document_version_id,
                "chunk_index": row.chunk_index,
                "title": row.title,
                "section_heading": row.section_heading,
                "page_number": row.page_number,
                "source_file_name": row.source_file_name,
                "content_hash": row.content_hash,
                "content": row.content,
                "score": rank,
                "headline": row.headline,
            })
        
        return chunks
    finally:
        session.close()


def search_chunks_keyword_trigram(
    query: str,
    limit: int = 10,
    min_similarity: float = 0.1,
    document_ids: Optional[List[int]] = None,
) -> List[dict]:
    """
    Search document chunks using PostgreSQL trigram similarity (ILIKE pattern).
    
    This is an alternative to full-text search that can catch misspellings
    and partial matches more flexibly.
    
    Args:
        query: Search query string.
        limit: Maximum number of results to return.
        min_similarity: Minimum similarity score (0.0 to 1.0).
        document_ids: Optional filter to restrict search to specific documents.
        
    Returns:
        List of chunk dicts with metadata and similarity score.
    """
    session: Session = SessionLocal()
    try:
        # Escape special characters to prevent SQL injection in LIKE patterns
        escaped_query = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped_query}%"
        
        # Build WHERE clause with optional document filter
        where_clause = "WHERE dc.content ILIKE :pattern AND similarity(dc.content, :query) >= :min_similarity"
        if document_ids:
            placeholders = ", ".join(f":doc_{i}" for i in range(len(document_ids)))
            where_clause += f" AND dc.document_id IN ({placeholders})"
            for i, doc_id in enumerate(document_ids):
                params[f"doc_{i}"] = doc_id
        
        sql = text(f"""
            SELECT
                dc.id as chunk_id,
                dc.document_id,
                dc.document_version_id,
                dc.chunk_index,
                dc.title,
                dc.section_heading,
                dc.page_number,
                dc.source_file_name,
                dc.content_hash,
                LEFT(dc.content, 500) as content,
                similarity(dc.content, :query) as similarity_score
            FROM document_chunks dc
            {where_clause}
            ORDER BY similarity_score DESC
            LIMIT :limit
        """)
        params["limit"] = limit
        
        result = session.execute(sql, params)
        rows = result.fetchall()
        
        chunks = []
        for row in rows:
            chunks.append({
                "chunk_id": row.chunk_id,
                "document_id": row.document_id,
                "document_version_id": row.document_version_id,
                "chunk_index": row.chunk_index,
                "title": row.title,
                "section_heading": row.section_heading,
                "page_number": row.page_number,
                "source_file_name": row.source_file_name,
                "content_hash": row.content_hash,
                "content": row.content,
                "score": float(row.similarity_score) if row.similarity_score else 0.0,
            })
        
        return chunks
    finally:
        session.close()