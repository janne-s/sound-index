#!/usr/bin/env python3
"""
Sound Index
Single-file app: DB, parser registry, ALS parser, LOGIC stub, Finder comments, GUI.
Designed for macOS (Finder comments via AppleScript / osascript).
"""

import sys
import os
import re
import gzip
import csv
import sqlite3
import xml.etree.ElementTree as ET
import plistlib
import subprocess
from pathlib import Path
from datetime import datetime
from threading import Lock
from contextlib import contextmanager

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QMessageBox,
    QLineEdit, QCheckBox, QTextEdit, QPushButton, QFrame, QTableWidget,
    QTableWidgetItem, QTabWidget, QHeaderView, QGroupBox, QSizePolicy,
    QFileDialog, QAbstractItemView, QMainWindow, QProgressBar, 
    QDialog, QDialogButtonBox, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QColor

# Constants
AUDIO_EXTENSIONS = (".aif", ".aiff", ".wav", ".caf", ".mp3", ".flac", ".ogg")
PROJECT_EXTENSIONS = (".als", ".logicx", ".logic")
DEFAULT_DB_NAME = "project_samples.db"
SEARCH_DEBOUNCE_MS = 450
PROGRESS_UPDATE_INTERVAL = 10

# ---------------------------
# DatabaseManager (project-type neutral)
# ---------------------------

class DatabaseManager:
    def __init__(self, db_path=None, pool_size=3):
        if db_path is None:
            db_path = Path.home() / DEFAULT_DB_NAME
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Main connection for initialization only
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.lock = Lock()
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.init_database()
        
        # Thread-safe connection pool
        self.pool = []
        self.pool_lock = Lock()
        self.pool_size = pool_size
        
        # Pre-populate pool
        for _ in range(pool_size):
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("PRAGMA foreign_keys = ON")
            self.pool.append(conn)

    def get_connection(self):
        """Get a connection from the pool or create a new one if pool is empty."""
        with self.pool_lock:
            if self.pool:
                return self.pool.pop()
        
        # Pool is empty, create temporary connection
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def return_connection(self, conn):
        """Return a connection to the pool if there's space, otherwise close it."""
        if conn is None:
            return
            
        with self.pool_lock:
            if len(self.pool) < self.pool_size:
                self.pool.append(conn)
            else:
                try:
                    conn.close()
                except Exception:
                    pass  # Connection already closed or invalid

    def close_all_connections(self):
        """Close all pooled connections. Call this on app shutdown."""
        with self.pool_lock:
            for conn in self.pool:
                try:
                    conn.close()
                except Exception:
                    pass
            self.pool.clear()
        
        try:
            self.conn.close()
        except Exception:
            pass

    def init_database(self):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL UNIQUE,
                    file_exists INTEGER DEFAULT 0,
                    comment_updated INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_checked TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sample_projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sample_id INTEGER NOT NULL,
                    project_file TEXT NOT NULL,
                    project_type TEXT NOT NULL,
                    UNIQUE(sample_id, project_file, project_type),
                    FOREIGN KEY(sample_id) REFERENCES samples(id) ON DELETE CASCADE
                )
            ''')

            # New table: track each project (file/path) and when it was last scanned + its mtime
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_file TEXT NOT NULL UNIQUE,
                    project_type TEXT,
                    last_scanned TEXT,
                    mtime REAL
                )
            ''')

            # Indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_path ON samples(file_path)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sample_projects_file ON sample_projects(project_file)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sample_projects_type ON sample_projects(project_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_projects_file ON projects(project_file)')

            self.conn.commit()

    def upsert_project_scan(self, project_file, project_type, scanned_at_iso, mtime):
        """Insert or update the projects table with last scanned time and mtime (float)."""
        with self.get_connection_context() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO projects (project_file, project_type, last_scanned, mtime)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(project_file) DO UPDATE SET
                    project_type=excluded.project_type,
                    last_scanned=excluded.last_scanned,
                    mtime=excluded.mtime
            ''', (project_file, project_type, scanned_at_iso, mtime))
            conn.commit()

    def get_project_record(self, project_file):
        """Return dict with keys project_file, project_type, last_scanned, mtime or None."""
        with self.get_connection_context() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT project_file, project_type, last_scanned, mtime FROM projects WHERE project_file = ?', (project_file,))
            r = cursor.fetchone()
            if not r:
                return None
            return {'project_file': r[0], 'project_type': r[1], 'last_scanned': r[2], 'mtime': r[3]}

    def get_all_projects(self):
        """Return list of (project_file, project_type) for all known projects."""
        with self.get_connection_context() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT project_file, project_type FROM projects')
            return cursor.fetchall()

    def add_sample(self, file_path, project_file, file_exists, comment_updated, project_type):
        """Add sample and associate it with a project file/type."""
        if not file_path:
            return
        
        with self.get_connection_context() as conn:
            cursor = conn.cursor()
            
            # Ensure sample row exists
            cursor.execute('SELECT id FROM samples WHERE file_path = ?', (file_path,))
            r = cursor.fetchone()
            if r:
                sample_id = r[0]
                cursor.execute('''
                    UPDATE samples
                    SET file_exists=?, comment_updated=?, last_checked=?
                    WHERE id=?
                ''', (int(bool(file_exists)), int(bool(comment_updated)), datetime.now().isoformat(), sample_id))
            else:
                cursor.execute('''
                    INSERT INTO samples (file_path, file_exists, comment_updated, last_checked)
                    VALUES (?, ?, ?, ?)
                ''', (file_path, int(bool(file_exists)), int(bool(comment_updated)), datetime.now().isoformat()))
                sample_id = cursor.lastrowid
            
            # Add project association
            if project_file:
                cursor.execute('''
                    INSERT OR IGNORE INTO sample_projects (sample_id, project_file, project_type)
                    VALUES (?, ?, ?)
                ''', (sample_id, project_file, project_type))
            
            conn.commit()

    def get_all_samples(self, limit=None, offset=0, search_term=None, project_filter=None):
        with self.lock:
            cursor = self.conn.cursor()
            query = '''
                SELECT s.id,
                       s.file_path,
                       COALESCE(GROUP_CONCAT(sp.project_type || ':' || sp.project_file, ' | '), '') AS project_files,
                       s.file_exists, s.comment_updated
                FROM samples s
                LEFT JOIN sample_projects sp ON sp.sample_id = s.id
            '''
            params = []
            conditions = []
            if search_term:
                conditions.append('(s.file_path LIKE ? OR sp.project_file LIKE ? OR sp.project_type LIKE ?)')
                params.extend([f'%{search_term}%', f'%{search_term}%', f'%{search_term}%'])
            if project_filter:
                conditions.append('(sp.project_file LIKE ? OR sp.project_type LIKE ?)')
                params.extend([f'%{project_filter}%', f'%{project_filter}%'])
            if conditions:
                query += ' WHERE ' + ' AND '.join(conditions)
            query += ' GROUP BY s.id ORDER BY s.file_path ASC'
            if limit:
                query += ' LIMIT ? OFFSET ?'
                params.extend([limit, offset])
            cursor.execute(query, params)
            rows = cursor.fetchall()

            # --- FIXED PROJECT DISPLAY ---
            cleaned_rows = []
            for sid, fpath, projects, exists, commented in rows:
                display_projects = ""
                if projects:
                    by_type = {}
                    for entry in projects.split(" | "):
                        if ":" not in entry:
                            continue
                        ptype, pfile = entry.split(":", 1)
                        fname = Path(pfile).name
                        by_type.setdefault(ptype, []).append(fname)
                    parts = []
                    for ptype, files in by_type.items():
                        parts.append(f"{ptype}: {', '.join(sorted(set(files)))}")
                    display_projects = " | ".join(parts)
                cleaned_rows.append((sid, fpath, display_projects, exists, commented))
            return cleaned_rows

    def get_stats(self):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM samples')
            total_samples = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM samples WHERE file_exists = 1')
            existing_files = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM samples WHERE comment_updated = 1')
            commented_files = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(DISTINCT project_file) FROM sample_projects')
            unique_project_files = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(DISTINCT project_type) FROM sample_projects')
            unique_project_types = cursor.fetchone()[0]

            return {
                'total_samples': total_samples,
                'existing_files': existing_files,
                'commented_files': commented_files,
                'unique_project_files': unique_project_files,
                'unique_project_types': unique_project_types
            }

    def get_project_count_for_sample(self, file_path):
        """Return dict: { project_type: count } for a sample path"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT project_type, COUNT(DISTINCT project_file)
                FROM samples s
                LEFT JOIN sample_projects sp ON sp.sample_id = s.id
                WHERE s.file_path = ?
                GROUP BY project_type
            ''', (file_path,))
            rows = cursor.fetchall()
            return {ptype: count for ptype, count in rows}

    def delete_sample_by_id(self, sample_id):
        with self.lock:
            cursor = self.conn.cursor()
            
            # Get project files associated with this sample before deleting
            cursor.execute("SELECT DISTINCT project_file FROM sample_projects WHERE sample_id = ?", (sample_id,))
            affected_projects = [row[0] for row in cursor.fetchall()]
            
            cursor.execute("DELETE FROM sample_projects WHERE sample_id = ?", (sample_id,))
            cursor.execute("DELETE FROM samples WHERE id = ?", (sample_id,))
            
            # Clean up projects that no longer have any samples
            for project_file in affected_projects:
                cursor.execute("SELECT COUNT(*) FROM sample_projects WHERE project_file = ?", (project_file,))
                if cursor.fetchone()[0] == 0:  # No more samples for this project
                    cursor.execute("DELETE FROM projects WHERE project_file = ?", (project_file,))
            
            self.conn.commit()

    def remove_samples_by_project(self, project_file):
        """Remove all samples associated with a specific project file AND the project record itself."""
        with self.lock:
            cursor = self.conn.cursor()
            # Get sample IDs that are associated with this project
            cursor.execute('''
                SELECT DISTINCT sample_id FROM sample_projects 
                WHERE project_file = ?
            ''', (project_file,))
            sample_ids = [row[0] for row in cursor.fetchall()]
            
            if sample_ids:
                # Remove project associations first
                cursor.execute('DELETE FROM sample_projects WHERE project_file = ?', (project_file,))
                
                # Remove samples that are no longer associated with any project
                placeholders = ','.join('?' * len(sample_ids))
                cursor.execute(f'''
                    DELETE FROM samples 
                    WHERE id IN ({placeholders}) 
                    AND id NOT IN (SELECT DISTINCT sample_id FROM sample_projects)
                ''', sample_ids)
            
            # NEW: Also remove the project record from the projects table
            cursor.execute('DELETE FROM projects WHERE project_file = ?', (project_file,))
            
            self.conn.commit()
            return len(sample_ids)

    def remove_projects_by_folder_pattern(self, folder_patterns):
        """
        Remove project records and their associations based on folder patterns.
        folder_patterns: list of strings to match against project paths (case-insensitive)
        Returns: (removed_projects_count, orphaned_samples_count)
        """
        if not folder_patterns:
            return 0, 0
            
        with self.lock:
            cursor = self.conn.cursor()
            
            # Build the WHERE clause for pattern matching
            conditions = []
            params = []
            for pattern in folder_patterns:
                conditions.append("LOWER(project_file) LIKE ?")
                params.append(f"%{pattern.lower()}%")
            
            where_clause = " OR ".join(conditions)
            
            # Get project files that will be removed
            cursor.execute(f"SELECT project_file FROM projects WHERE {where_clause}", params)
            projects_to_remove = [row[0] for row in cursor.fetchall()]
            
            if not projects_to_remove:
                return 0, 0
            
            # Count samples that will become orphaned (not associated with any remaining projects)
            project_placeholders = ','.join('?' * len(projects_to_remove))
            cursor.execute(f'''
                SELECT COUNT(DISTINCT s.id)
                FROM samples s
                JOIN sample_projects sp ON sp.sample_id = s.id
                WHERE sp.project_file IN ({project_placeholders})
                AND s.id NOT IN (
                    SELECT DISTINCT sp2.sample_id 
                    FROM sample_projects sp2 
                    WHERE sp2.project_file NOT IN ({project_placeholders})
                )
            ''', projects_to_remove + projects_to_remove)
            orphaned_samples = cursor.fetchone()[0]
            
            # Remove project associations
            cursor.execute(f"DELETE FROM sample_projects WHERE project_file IN ({project_placeholders})", projects_to_remove)
            
            # Remove project records
            cursor.execute(f"DELETE FROM projects WHERE project_file IN ({project_placeholders})", projects_to_remove)
            
            self.conn.commit()
            return len(projects_to_remove), orphaned_samples

    def delete_samples_batch(self, sample_ids):
        """Delete multiple samples efficiently and clean up orphaned projects."""
        if not sample_ids:
            return
            
        with self.lock:
            cursor = self.conn.cursor()
            
            # Get all affected project files before deletion
            placeholders = ','.join('?' * len(sample_ids))
            cursor.execute(f"SELECT DISTINCT project_file FROM sample_projects WHERE sample_id IN ({placeholders})", sample_ids)
            affected_projects = [row[0] for row in cursor.fetchall()]
            
            # Delete sample associations and samples
            cursor.execute(f"DELETE FROM sample_projects WHERE sample_id IN ({placeholders})", sample_ids)
            cursor.execute(f"DELETE FROM samples WHERE id IN ({placeholders})", sample_ids)
            
            # Clean up orphaned projects
            for project_file in affected_projects:
                cursor.execute("SELECT COUNT(*) FROM sample_projects WHERE project_file = ?", (project_file,))
                if cursor.fetchone()[0] == 0:  # No more samples for this project
                    cursor.execute("DELETE FROM projects WHERE project_file = ?", (project_file,))
            
            self.conn.commit()
            return len(affected_projects)

    def get_projects_by_folder_preview(self, folder_patterns):
        """Preview which projects would be affected by folder pattern cleanup."""
        if not folder_patterns:
            return []
            
        with self.lock:
            cursor = self.conn.cursor()
            conditions = []
            params = []
            for pattern in folder_patterns:
                conditions.append("LOWER(project_file) LIKE ?")
                params.append(f"%{pattern.lower()}%")
            
            where_clause = " OR ".join(conditions)
            cursor.execute(f"SELECT project_file, project_type FROM projects WHERE {where_clause}", params)
            return cursor.fetchall()

    def sync_comment_status(self, file_path):
        """Check Finder comment for any TYPE: N projects marker and update sample row."""
        exists = os.path.exists(file_path)
        comment_text = get_file_comment(file_path) if exists else ""
        has_marker = bool(re.search(r'\b[A-Z]+:\s*\d+\s+projects', comment_text))
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE samples
                SET file_exists=?, comment_updated=?, last_checked=?
                WHERE file_path=?
            """, (int(bool(exists)), int(bool(has_marker)), datetime.now().isoformat(), file_path))
            if cursor.rowcount == 0:
                # insert missing sample row
                cursor.execute("""
                    INSERT INTO samples (file_path, file_exists, comment_updated, last_checked)
                    VALUES (?, ?, ?, ?)
                """, (file_path, int(bool(exists)), int(bool(has_marker)), datetime.now().isoformat()))
            self.conn.commit()

    @contextmanager
    def get_connection_context(self):
        """Context manager for safe connection handling."""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.return_connection(conn)

def compute_project_mtime(path, max_files=1000):
    """
    Compute project mtime with memory optimization.
    For large directories, sample files instead of walking everything.
    """
    try:
        if os.path.isfile(path):
            return os.path.getmtime(path)
        elif os.path.isdir(path):
            max_mtime = 0.0
            file_count = 0
            
            for root, dirs, files in os.walk(path):
                # Skip hidden directories and common build/cache dirs
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'node_modules'}]
                
                for fn in files:
                    if file_count >= max_files:
                        break
                    try:
                        p = os.path.join(root, fn)
                        mt = os.path.getmtime(p)
                        if mt > max_mtime:
                            max_mtime = mt
                        file_count += 1
                    except Exception:
                        continue
                        
                if file_count >= max_files:
                    break
            
            # Fallback to directory mtime if no files found
            if max_mtime == 0.0:
                return os.path.getmtime(path)
            return max_mtime
    except Exception:
        return 0.0

# ---------------------------
# Finder comment helpers (macOS via osascript)
# ---------------------------

def get_file_comment(file_path):
    """Get Finder comment using AppleScript (osascript). Returns empty string on failure."""
    try:
        abs_path = os.path.abspath(file_path)
        applescript = f'''
        tell application "Finder"
            get comment of (POSIX file "{abs_path}" as alias)
        end tell
        '''
        result = subprocess.run(['osascript', '-e', applescript], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return ""

def set_file_comment(file_path, comment):
    """Set Finder comment using AppleScript (osascript). Returns True on success."""
    if not os.path.exists(file_path):
        return False
        
    try:
        abs_path = os.path.abspath(file_path)
        # Escape special characters properly
        escaped = (comment.replace('\\', '\\\\')
                         .replace('"', '\\"')
                         .replace('\n', '\\n')
                         .replace('\r', '\\r'))
        applescript = f'''
        tell application "Finder"
            set comment of (POSIX file "{abs_path}" as alias) to "{escaped}"
        end tell
        '''
        subprocess.run(['osascript', '-e', applescript], 
                      capture_output=True, text=True, 
                      check=True, timeout=10)
        return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"set_file_comment failed for {file_path}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error in set_file_comment for {file_path}: {e}")
        return False

def append_to_file_comment(file_path, new_fragment):
    """Append fragment like 'ALS: 3 projects' to Finder comment if not already present."""
    try:
        existing = get_file_comment(file_path) or ""
        existing = existing.strip()
        if existing:
            if new_fragment in existing:
                return False
            combined = f"{existing} | {new_fragment}"
        else:
            combined = new_fragment
        return set_file_comment(file_path, combined)
    except Exception as e:
        print(f"append_to_file_comment error: {e}")
        return False

def clear_project_markers_from_comment(comment_text):
    """Remove TYPE: N projects fragments (and separators) from comment text."""
    if not comment_text:
        return ""
    # Remove occurrences like " | ALS: 3 projects" OR "ALS: 3 projects"
    new = re.sub(r'(?:\s*\|\s*)?\b[A-Z]+:\s*\d+\s+projects', '', comment_text)
    # Clean trailing separators and whitespace
    new = re.sub(r'\s*\|\s*$', '', new).strip()
    return new

# ---------------------------
# Parser registry & process_project()
# ---------------------------

PROJECT_PARSERS = {}

def register_project_parser(project_type):
    def decorator(func):
        PROJECT_PARSERS[project_type] = func
        return func
    return decorator

def detect_project_type(path: str) -> str:
    # Strip trailing slashes for directory bundles
    clean_path = path.rstrip('/').lower()
    if clean_path.endswith(".als"):
        return "ALS"
    if clean_path.endswith((".logic", ".logicx")):
        return "LOGIC"
    return "UNKNOWN"

def process_project(file_path, db_manager: DatabaseManager,
                    update_comments=False, progress_callback=None,
                    skip_folder_fn=None, skip_sample_fn=None, force=True, cleanup_existing=False):
    """Unified processing entrypoint. If force is False, use project's stored mtime to skip unchanged projects.
       If cleanup_existing is True, remove existing entries for this project first.
       skip_sample_fn: optional function to filter individual sample paths.
       Returns (num_samples_processed, updated_comments_count).
    """
    project_type = detect_project_type(file_path)
    parser = PROJECT_PARSERS.get(project_type)
    if not parser:
        raise ValueError(f"No parser registered for project type '{project_type}'")

    # Clean up existing entries if requested
    if cleanup_existing and db_manager:
        removed_count = db_manager.remove_samples_by_project(file_path)
        if progress_callback and removed_count > 0:
            progress_callback(f"Removed {removed_count} existing samples from {Path(file_path).name}")

    # mtime check (incremental)
    try:
        current_mtime = compute_project_mtime(file_path)
    except Exception:
        current_mtime = 0.0

    if db_manager and not force and not cleanup_existing:
        rec = db_manager.get_project_record(file_path)
        if rec and rec.get('mtime') == current_mtime:
            # no change -> skip
            if progress_callback:
                progress_callback(f"Skipped (no changes): {Path(file_path).name}")
            return 0, 0

    samples = parser(file_path)

    # Apply folder filtering if provided
    if skip_folder_fn:
        samples = [s for s in samples if not skip_folder_fn(Path(s).parent)]

    # Apply sample path filtering if provided
    if skip_sample_fn:
        original_count = len(samples)
        samples = [s for s in samples if not skip_sample_fn(s)]
        filtered_count = original_count - len(samples)
        if filtered_count > 0 and progress_callback:
            progress_callback(f"Filtered out {filtered_count} samples from {Path(file_path).name}")

    updated_count = 0
    for i, sample_path in enumerate(sorted(samples), 1):
        file_exists = os.path.exists(sample_path)
        db_manager.add_sample(sample_path, file_path, file_exists, False, project_type)

        if update_comments and file_exists:
            counts = db_manager.get_project_count_for_sample(sample_path)
            for ptype, count in counts.items():
                fragment = f"{ptype}: {count} projects"
                if append_to_file_comment(sample_path, fragment):
                    updated_count += 1
            db_manager.sync_comment_status(sample_path)

        if progress_callback and (i % PROGRESS_UPDATE_INTERVAL == 0 or i == len(samples)):
            progress_callback(f"Processed {i}/{len(samples)} samples from {Path(file_path).name}")

    # record project scan (mtime + timestamp)
    if db_manager:
        try:
            db_manager.upsert_project_scan(file_path, project_type, datetime.now().isoformat(), current_mtime)
        except Exception:
            pass

    return len(samples), updated_count

# ---------------------------
# ALS parser (robust extraction based on user's original logic)
# ---------------------------

@register_project_parser("ALS")
def parse_als(als_file_path: str):
    """
    Parse Ableton .als (gzipped XML) project and extract referenced sample paths.
    This mirrors the original extraction logic with multiple fallback strategies.
    Returns: list of sample paths (strings)
    """
    sample_entries = set()

    try:
        with gzip.open(als_file_path, "rb") as f:
            xml_content = f.read()
    except Exception:
        # Not gzipped? try reading raw
        try:
            with open(als_file_path, "rb") as f:
                xml_content = f.read()
        except Exception:
            return []

    try:
        root = ET.fromstring(xml_content)
    except Exception:
        return []

    for sample_ref in root.findall(".//SampleRef"):
        file_ref = sample_ref.find("./FileRef")
        if file_ref is None:
            continue

        # Name element (may hold filename)
        name_elem = file_ref.find("./Name[@Value]")
        filename = name_elem.attrib["Value"] if name_elem is not None and "Value" in name_elem.attrib else None

        candidate = None

        # 1) Path element with Value attribute
        path_elem = file_ref.find("./Path")
        if path_elem is not None and path_elem.attrib.get("Value"):
            candidate = path_elem.attrib["Value"].strip()

        # 2) AbsolutePath element text
        if candidate is None:
            abs_path_elem = file_ref.find("./AbsolutePath")
            if abs_path_elem is not None and abs_path_elem.text:
                candidate = abs_path_elem.text.strip()

        # 3) SearchHint / PathHint with RelativePathElement sequence
        if candidate is None:
            path_hint = file_ref.find("./SearchHint/PathHint")
            if path_hint is not None:
                parts = [e.attrib.get("Dir") for e in path_hint.findall("./RelativePathElement")]
                parts = [p for p in parts if p]
                if parts:
                    try:
                        base = Path("/", *parts)
                        candidate = str(base / filename) if filename else str(base)
                    except Exception:
                        candidate = None

        # 4) RelativePath attributes or elements
        if candidate is None:
            rel_attr = file_ref.find("./RelativePath")
            if rel_attr is not None and "Value" in rel_attr.attrib:
                rel_value = rel_attr.attrib["Value"].strip()
                if filename and not rel_value.endswith(filename):
                    candidate = str(Path(rel_value) / filename)
                else:
                    candidate = rel_value
            elif rel_attr is not None:
                parts = [e.attrib.get("Dir") for e in rel_attr.findall("./RelativePathElement")]
                parts = [p for p in parts if p]
                if filename and parts:
                    candidate = str(Path(*parts) / filename)
                elif filename:
                    candidate = filename

        # 5) Fallback: Name only
        if candidate is None and filename:
            candidate = filename

        # Add candidate if non-empty
        if candidate:
            sample_entries.add(candidate)

    return sorted(sample_entries)

# ---------------------------
# LOGIC parser
# ---------------------------

@register_project_parser("LOGIC")
def parse_logicx(logicx_path: str):
    """
    Extract sample references from Logic Pro .logic/.logicx project.
    Handles both legacy .logic (documentData) and modern .logicx (MetaData.plist) formats.
    Returns list of sample paths (strings).
    """
    import plistlib
    import re
    
    # Strip trailing slashes that can come from drag/drop of directories
    clean_path = logicx_path.rstrip('/')
    
    AUDIO_EXTS = (".aif", ".aiff", ".wav", ".caf", ".mp3", ".flac", ".m4a")
    AUDIO_KEYS = [
        "AudioFiles",
        "SamplerInstrumentsFiles", 
        "ImpulsResponsesFiles",
        "UnusedAudioFiles",
        "PlaybackFiles",
        "QuicksamplerFiles",
        "UltrabeatFiles",
    ]
    
    def find_document_data(project_path):
        """Return path to documentData inside a .logic project."""
        for root, _, files in os.walk(project_path):
            for f in files:
                if f == "documentData":
                    return os.path.join(root, f)
        raise FileNotFoundError("documentData not found in project")

    def extract_audio_from_document_data(document_data_path):
        """Scan binary documentData for ASCII strings with audio file references."""
        try:
            with open(document_data_path, "rb") as f:
                data = f.read()
        except Exception:
            return []

        strings_found = re.findall(rb"[ -~]{4,}", data)
        audio_files = []
        folders = []
        prev = None

        for s in strings_found:
            decoded = s.decode(errors="ignore")
            if prev and prev.lower().endswith(AUDIO_EXTS) and decoded.startswith("/"):
                folders.append(decoded)
                audio_files.append(os.path.join(decoded, prev))
            prev = decoded

        return audio_files

    def find_metadata_plist(project_path):
        """Return path to MetaData.plist inside a .logicx project."""
        for root, _, files in os.walk(project_path):
            for f in files:
                if f == "MetaData.plist":
                    return os.path.join(root, f)
        raise FileNotFoundError("MetaData.plist not found in project")

    def extract_audio_from_metadata(plist_path, project_root):
        """Extract audio file references from MetaData.plist."""
        try:
            with open(plist_path, "rb") as f:
                plist_data = plistlib.load(f)
        except Exception:
            return []

        audio_files = []

        for key in AUDIO_KEYS:
            if key not in plist_data:
                continue
            file_list = plist_data[key]
            if not isinstance(file_list, list):
                continue
                
            for entry in file_list:
                if not isinstance(entry, str):
                    continue
                    
                if entry.startswith("/"):  # absolute path
                    audio_files.append(entry)
                else:  # relative path - check both bundled and external locations
                    # Option 1: Bundled inside .logicx/Media/
                    bundled_path = os.path.join(project_root, "Media", entry)
                    
                    # Option 2: External Audio Files folder (sibling to .logicx)
                    project_parent = os.path.dirname(project_root)
                    external_path = os.path.join(project_parent, entry)
                    
                    # Use whichever path actually exists, prefer bundled
                    if os.path.exists(bundled_path):
                        audio_files.append(os.path.normpath(bundled_path))
                    elif os.path.exists(external_path):
                        audio_files.append(os.path.normpath(external_path))
                    else:
                        # Neither exists, but include the bundled path as default
                        # (might be a missing file that user wants to track)
                        audio_files.append(os.path.normpath(bundled_path))

        return audio_files

    # Main parsing logic
    entries = set()
    p = Path(logicx_path)
    if not p.exists():
        return []

    try:
        # Use clean_path (without trailing slash) for extension checking
        if clean_path.lower().endswith(".logic"):
            # Legacy .logic format
            doc_path = find_document_data(logicx_path)
            results = extract_audio_from_document_data(doc_path)
            entries.update(results)
        elif clean_path.lower().endswith(".logicx"):
            # Modern .logicx format
            plist_path = find_metadata_plist(logicx_path)
            # Pass the project root (logicx_path) as the base directory
            results = extract_audio_from_metadata(plist_path, logicx_path)
            entries.update(results)
        else:
            # Fallback: treat as bundle and scan for audio files
            audio_dirs = ["Audio Files", "Samples", "Imported", "Audio"]
            for ad in audio_dirs:
                d = p / ad
                if d.exists() and d.is_dir():
                    for f in d.rglob("*"):
                        if f.suffix.lower() in AUDIO_EXTENSIONS:
                            entries.add(str(f.resolve()))
    except FileNotFoundError:
        # If structured parsing fails, fallback to scanning the bundle
        if p.is_dir():
            for f in p.rglob("*"):
                if f.suffix.lower() in AUDIO_EXTENSIONS:
                    entries.add(str(f.resolve()))

    return sorted(entries)

# ---------------------------
# Threads: Processing, Comment Writing, Clearing
# ---------------------------

class ProcessingThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str, int, int)  # message, total_files, updated_count

    def __init__(self, project_file, db_manager, update_comments=False, skip_folder_fn=None, skip_sample_fn=None, cleanup_existing=False):
        super().__init__()
        self.project_file = project_file
        self.db_manager = db_manager
        self.update_comments = update_comments
        self.skip_folder_fn = skip_folder_fn
        self.skip_sample_fn = skip_sample_fn
        self.cleanup_existing = cleanup_existing
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self._cancelled:
                self.finished_signal.emit("Operation cancelled", 0, 0)
                return
                
            # default to True (force) if attribute not present to preserve prior behavior
            force_flag = getattr(self, 'force', True)
            total_files, updated_count = process_project(
                self.project_file,
                self.db_manager,
                update_comments=self.update_comments,
                progress_callback=self.progress_signal.emit,
                skip_folder_fn=self.skip_folder_fn,
                skip_sample_fn=self.skip_sample_fn,
                force=force_flag,
                cleanup_existing=self.cleanup_existing
            )
            
            if self._cancelled:
                self.finished_signal.emit("Operation cancelled", 0, 0)
                return
                
            message = f"Processed {total_files} samples from {Path(self.project_file).name}"
            if self.update_comments:
                message += f"\nUpdated comments for {updated_count} files"
            self.finished_signal.emit(message, total_files, updated_count)
        except Exception as e:
            if self._cancelled:
                self.finished_signal.emit("Operation cancelled", 0, 0)
            else:
                self.finished_signal.emit(f"Error: {e}", 0, 0)


class CommentWritingThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int, int)  # updated_count, total_count

    def __init__(self, samples, db_manager):
        super().__init__()
        self.samples = samples
        self.db_manager = db_manager

    def run(self):
        updated_count = 0
        total = len(self.samples)
        for i, sample in enumerate(self.samples, 1):
            sid, file_path, project_files, file_exists, comment_updated = sample
            if not file_exists:
                continue
            self.progress_signal.emit(f"Processing {i}/{total}: {Path(file_path).name}")
            try:
                counts = self.db_manager.get_project_count_for_sample(file_path)
                for ptype, cnt in counts.items():
                    frag = f"{ptype}: {cnt} projects"
                    if append_to_file_comment(file_path, frag):
                        updated_count += 1
                self.db_manager.sync_comment_status(file_path)
            except Exception as e:
                print(f"Failed to update comment for {file_path}: {e}")
                continue
        self.finished_signal.emit(updated_count, total)

class CommentClearingThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int, int)  # cleared_count, total_count

    def __init__(self, samples, db_manager):
        super().__init__()
        self.samples = samples
        self.db_manager = db_manager

    def run(self):
        cleared = 0
        total = len(self.samples)
        for i, sample in enumerate(self.samples, 1):
            sid, file_path, project_files, file_exists, comment_updated = sample
            if not file_exists:
                continue
            self.progress_signal.emit(f"Clearing {i}/{total}: {Path(file_path).name}")
            try:
                existing = get_file_comment(file_path) or ""
                new = clear_project_markers_from_comment(existing)
                if new != (existing or ""):
                    if set_file_comment(file_path, new):
                        cleared += 1
                        self.db_manager.sync_comment_status(file_path)
            except Exception as e:
                print(f"Failed to clear comment for {file_path}: {e}")
                continue
        self.finished_signal.emit(cleared, total)

# ---------------------------
# GUI: DatabaseViewer
# ---------------------------

class DeletableTableWidget(QTableWidget):
    def __init__(self, parent_view=None):
        super().__init__(parent=parent_view)
        self.parent_view = parent_view

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.parent_view:
                self.parent_view.delete_selected_entries()  # Press “Delete Selected”
        elif event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_A:
            total_rows = self.rowCount()
            selected_rows = len(set(idx.row() for idx in self.selectedIndexes()))
            if selected_rows == total_rows:
                self.clearSelection()
            else:
                self.selectAll()
        else:
            super().keyPressEvent(event)

class DatabaseViewer(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.current_samples = []
        self.comment_thread = None
        self.init_ui()
        self.refresh_data()

    def init_ui(self):
        layout = QVBoxLayout()

        # Controls
        controls_group = QGroupBox("")
        controls_layout = QVBoxLayout()

        first_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search file paths or names...")
        self.search_input.textChanged.connect(self.on_search_timer)
        first_row.addWidget(QLabel("Search:"))
        first_row.addWidget(self.search_input)

        self.project_filter = QLineEdit()
        self.project_filter.setPlaceholderText("Filter by project file or type...")
        self.project_filter.textChanged.connect(self.on_search_timer)
        first_row.addWidget(QLabel("Project Filter:"))
        first_row.addWidget(self.project_filter)

        controls_layout.addLayout(first_row)

        #Second row
        second_row = QHBoxLayout()
        self.export_btn = QPushButton("Export to CSV")
        self.export_btn.clicked.connect(self.export_to_csv)
        second_row.addWidget(self.export_btn)

        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.setToolTip("Delete selected rows from the database")
        self.delete_btn.clicked.connect(self.delete_selected_entries)
        second_row.addWidget(self.delete_btn)

        self.comment_btn = QPushButton("Add Comments")
        self.comment_btn.setToolTip("Append project names to Finder comments for selected (or all visible) files")
        self.comment_btn.clicked.connect(self.on_add_comments)
        second_row.addWidget(self.comment_btn)

        self.clear_btn = QPushButton("Clear Comments")
        self.clear_btn.setToolTip("Remove Finder comments for selected (or all visible) files")
        self.clear_btn.clicked.connect(self.on_clear_comments)
        second_row.addWidget(self.clear_btn)

        # In the second_row layout, add this button:
        self.cleanup_projects_btn = QPushButton("Clean Up Projects…")
        self.cleanup_projects_btn.setToolTip("Remove project references based on folder patterns")
        self.cleanup_projects_btn.clicked.connect(self.on_cleanup_projects_by_folder)
        second_row.addWidget(self.cleanup_projects_btn)

        second_row.addStretch()
        controls_layout.addLayout(second_row)
        controls_group.setLayout(controls_layout)

        layout.addWidget(controls_group)

        # Stats
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("font-weight:bold; padding:8px;")
        layout.addWidget(self.stats_label)

        # Table
        self.table = DeletableTableWidget(parent_view=self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['File Name', 'File Path', 'Project Files', 'File Exists', 'Comment Updated'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSortingEnabled(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        layout.addWidget(self.table)

        # Timer for search debounce
        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self.refresh_data)

        self.setLayout(layout)

    def on_search_timer(self):
        self.search_timer.start(SEARCH_DEBOUNCE_MS)

    def refresh_data(self):
        project_filter = self.project_filter.text().strip() or None
        search_term = self.search_input.text().strip() or None

        samples = self.db_manager.get_all_samples(limit=10000, search_term=search_term, project_filter=project_filter)
        self.current_samples = samples
        self.table.setRowCount(len(samples))
        
        for row, sample in enumerate(samples):
            sid, file_path, project_files, file_exists, comment_updated = sample
            
            # Create table items
            name_item = QTableWidgetItem(Path(file_path).name)
            path_item = QTableWidgetItem(file_path)
            projects_item = QTableWidgetItem(project_files)
            exists_item = QTableWidgetItem('Yes' if file_exists else 'No')
            comment_item = QTableWidgetItem('Yes' if comment_updated else 'No')
            
            # Apply styling for missing files
            if not file_exists:
                # Light gray color for missing files
                missing_color = QColor(200, 140, 140)
                name_item.setForeground(missing_color)
                path_item.setForeground(missing_color)
                #projects_item.setForeground(missing_color)
                exists_item.setForeground(missing_color)
                #comment_item.setForeground(missing_color)
            
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, path_item)
            self.table.setItem(row, 2, projects_item)
            self.table.setItem(row, 3, exists_item)
            self.table.setItem(row, 4, comment_item)

        stats = self.db_manager.get_stats()
        stats_text = (f"Total Samples: {stats['total_samples']} | "
                      f"Existing: {stats['existing_files']} | "
                      f"Commented: {stats['commented_files']} | "
                      f"Unique Projects: {stats['unique_project_files']} | "
                      f"Project Types: {stats['unique_project_types']}")
        self.stats_label.setText(stats_text)

    def export_to_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export to CSV", str(Path.home() / "Desktop" / "project_samples_export.csv"), "CSV files (*.csv)")
        if not file_path:
            return
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['File Name', 'File Path', 'Project Files', 'File Exists', 'Comment Updated'])
                for sid, file_path_text, project_files, file_exists, comment_updated in self.current_samples:
                    writer.writerow([Path(file_path_text).name, file_path_text, project_files, 'Yes' if file_exists else 'No', 'Yes' if comment_updated else 'No'])
            QMessageBox.information(self, "Export Complete", f"Exported {len(self.current_samples)} rows to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    def delete_selected_entries(self):
        selected_items = self.table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select at least one entry to delete.")
            return
        
        # Get unique row indices from selected items
        rows = sorted({item.row() for item in selected_items})
        ids_to_delete = [self.current_samples[r][0] for r in rows]
        
        confirm = QMessageBox.question(
            self, "Confirm Deletion", 
            f"Are you sure you want to delete {len(ids_to_delete)} entries from the database?", 
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm != QMessageBox.Yes:
            return
        
        try:
            # Delete each sample individually to ensure proper cleanup
            for sid in ids_to_delete:
                self.db_manager.delete_sample_by_id(sid)
            
            # Additional cleanup: remove any remaining orphaned projects
            with self.db_manager.lock:
                cursor = self.db_manager.conn.cursor()
                cursor.execute('''
                    DELETE FROM projects 
                    WHERE project_file NOT IN (
                        SELECT DISTINCT project_file 
                        FROM sample_projects 
                        WHERE project_file IS NOT NULL
                    )
                ''')
                orphaned_count = cursor.rowcount
                self.db_manager.conn.commit()
            
            # Clear selection BEFORE refreshing to prevent stale row reference issues
            self.table.clearSelection()
            self.refresh_data()
            
            message = f"Deleted {len(ids_to_delete)} entries."
            if orphaned_count > 0:
                message += f" Cleaned up {orphaned_count} orphaned project records."
            
            QMessageBox.information(self, "Deleted", message)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete entries: {e}")

    def on_add_comments(self):
        if hasattr(self, 'comment_thread') and self.comment_thread is not None and self.comment_thread.isRunning():
            QMessageBox.information(self, "Already running", "Comment update is already in progress.")
            return
        selected_items = self.table.selectedItems()
        rows = sorted({item.row() for item in selected_items}) if selected_items else []
        if rows:
            samples = [self.current_samples[r] for r in rows]
        else:
            samples = list(self.current_samples)
        if not samples:
            QMessageBox.information(self, "No samples", "No samples to update.")
            return
        # Disable UI
        self.comment_btn.setEnabled(False); self.export_btn.setEnabled(False)
        self.comment_thread = CommentWritingThread(samples, self.db_manager)
        self.comment_thread.progress_signal.connect(self._handle_comment_progress)
        self.comment_thread.finished_signal.connect(self._handle_comment_finished)
        self.comment_thread.start()
        self.stats_label.setText("Starting comment update...")

    def on_clear_comments(self):
        if hasattr(self, 'comment_thread') and self.comment_thread is not None and self.comment_thread.isRunning():
            QMessageBox.information(self, "Already running", "Another operation is in progress.")
            return
        selected_items = self.table.selectedItems()
        rows = sorted({item.row() for item in selected_items}) if selected_items else []
        samples = [self.current_samples[r] for r in rows] if rows else list(self.current_samples)
        if not samples:
            QMessageBox.information(self, "No samples", "No samples to clear.")
            return
        self.comment_btn.setEnabled(False); self.export_btn.setEnabled(False)
        self.comment_thread = CommentClearingThread(samples, self.db_manager)
        self.comment_thread.progress_signal.connect(self._handle_comment_progress)
        self.comment_thread.finished_signal.connect(self._handle_clear_finished)
        self.comment_thread.start()
        self.stats_label.setText("Starting comment clear...")

    def _handle_comment_progress(self, message):
        self.stats_label.setText(message)

    def _handle_comment_finished(self, updated_count, total_count):
        self.comment_btn.setEnabled(True); self.export_btn.setEnabled(True)
        self.refresh_data()
        QMessageBox.information(self, "Comments Updated", f"Updated comments for {updated_count} out of {total_count} files.")
        self.stats_label.setText(f"Finished: updated {updated_count} of {total_count}")
        self.comment_thread = None

    def _handle_clear_finished(self, cleared_count, total_count):
        self.comment_btn.setEnabled(True); self.export_btn.setEnabled(True)
        self.refresh_data()
        QMessageBox.information(self, "Comments Cleared", f"Cleared comments for {cleared_count} out of {total_count} files.")
        self.stats_label.setText(f"Finished: cleared {cleared_count} of {total_count}")
        self.comment_thread = None

    
    def on_cleanup_projects_by_folder(self):
        """Show dialog to clean up projects by folder patterns."""
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QTextEdit, QPushButton, QListWidget, QListWidgetItem
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Clean Up Projects by Folder")
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(500)
        
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel("Enter folder patterns to remove project references (one per line).\n"
                             "Examples: backup, old, freeze, archive, tmp\n"
                             "This removes project associations but keeps sample records.")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Pattern input
        pattern_input = QTextEdit()
        pattern_input.setMaximumHeight(100)
        pattern_input.setPlaceholderText("backup\nold\nfreeze\narchive")
        layout.addWidget(QLabel("Folder patterns:"))
        layout.addWidget(pattern_input)
        
        # Preview area
        preview_list = QListWidget()
        layout.addWidget(QLabel("Preview (projects that would be removed):"))
        layout.addWidget(preview_list)
        
        # Preview button
        preview_btn = QPushButton("Preview")
        layout.addWidget(preview_btn)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)
    
        def update_preview():
            patterns = [p.strip() for p in pattern_input.toPlainText().split('\n') if p.strip()]
            preview_list.clear()
            
            if patterns:
                projects = self.db_manager.get_projects_by_folder_preview(patterns)
                for project_file, project_type in projects:
                    item_text = f"{project_type}: {project_file}"
                    preview_list.addItem(QListWidgetItem(item_text))
                
                if not projects:
                    preview_list.addItem(QListWidgetItem("No matching projects found"))
            else:
                preview_list.addItem(QListWidgetItem("Enter patterns to see preview"))
    
        preview_btn.clicked.connect(update_preview)
        pattern_input.textChanged.connect(lambda: QTimer.singleShot(300, update_preview))
        
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        dialog.setLayout(layout)
        
        # Show initial preview
        update_preview()
        
        if dialog.exec_() == QDialog.Accepted:
            patterns = [p.strip() for p in pattern_input.toPlainText().split('\n') if p.strip()]
            if not patterns:
                QMessageBox.warning(self, "No Patterns", "Please enter at least one folder pattern.")
                return
                
            # Confirm cleanup
            projects = self.db_manager.get_projects_by_folder_preview(patterns)
            if not projects:
                QMessageBox.information(self, "No Projects", "No projects match the specified patterns.")
                return
                
            reply = QMessageBox.question(self, "Confirm Cleanup",
                                       f"Remove {len(projects)} project references?\n\n"
                                       f"This will remove project associations but keep sample records.\n"
                                       f"Some samples may become orphaned (not associated with any projects).",
                                       QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                try:
                    removed_projects, orphaned_samples = self.db_manager.remove_projects_by_folder_pattern(patterns)
                    
                    self.refresh_data()
                    
                    message = f"Removed {removed_projects} project references."
                    if orphaned_samples > 0:
                        message += f"\n{orphaned_samples} samples are now orphaned (no project associations)."
                    
                    QMessageBox.information(self, "Cleanup Complete", message)
                    
                except Exception as e:
                    QMessageBox.critical(self, "Cleanup Failed", f"Error during cleanup: {e}")

# ---------------------------
# GUI: MainWindow (drag/drop + queue)
# ---------------------------

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sound Index")
        self.setGeometry(200, 200, 1000, 700)
        self.setAcceptDrops(True)

        self.db_manager = DatabaseManager()
        self.queue = []
        self.current_processing = None

        self.init_ui()

    def closeEvent(self, event):
        """Handle application shutdown."""
        # Cancel any running operations
        if self.current_processing and self.current_processing.isRunning():
            self.current_processing.cancel()
            self.current_processing.wait(3000)  # Wait up to 3 seconds
        
        # Close database connections
        if hasattr(self, 'db_manager'):
            self.db_manager.close_all_connections()
        
        event.accept()

    def init_ui(self):
        layout = QVBoxLayout()

        self.tabs = QTabWidget()

        # Process tab
        self.process_tab = QWidget()
        p_layout = QVBoxLayout()
        # Folder filtering
        filter_group = QGroupBox("")
        f_layout = QHBoxLayout()
        self.enable_folder_filter = QCheckBox("Enable folder filtering")
        self.folder_filter_input = QLineEdit()
        self.folder_filter_input.setPlaceholderText("backup, freeze, packs, tmp, archive, old (comma-separated)")
        f_layout.addWidget(self.enable_folder_filter)
        f_layout.addWidget(self.folder_filter_input)
        filter_group.setLayout(f_layout)
        p_layout.addWidget(filter_group)

        self.drop_label = QLabel("Drag & drop a project file or folder here")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet("font-size:16px; border:2px dashed #888; padding:40px;")
        p_layout.addWidget(self.drop_label)

        # Add scan buttons
        scan_buttons_layout = QHBoxLayout()
        
        self.scan_folder_btn = QPushButton("Scan Folder…")
        self.scan_folder_btn.setToolTip("Choose a folder and incrementally re-scan changed projects (respects folder filters).")
        self.scan_folder_btn.clicked.connect(self.on_scan_folder)
        scan_buttons_layout.addWidget(self.scan_folder_btn)
        
        self.full_rescan_btn = QPushButton("Full Re-scan")
        self.full_rescan_btn.setToolTip("Force re-scan of all known projects in the database (respects folder filters).")
        self.full_rescan_btn.clicked.connect(self.on_full_rescan)
        scan_buttons_layout.addWidget(self.full_rescan_btn)
        
        # Add cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip("Cancel current processing operation")
        self.cancel_btn.clicked.connect(self.on_cancel_processing)
        self.cancel_btn.setEnabled(False)
        scan_buttons_layout.addWidget(self.cancel_btn)
        
        scan_buttons_layout.addStretch()  # Push buttons to the left
        p_layout.addLayout(scan_buttons_layout)

        # Log
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        p_layout.addWidget(self.progress_text)

        self.process_tab.setLayout(p_layout)
        self.tabs.addTab(self.process_tab, "Process Files")

        # Database viewer tab
        self.db_viewer = DatabaseViewer(self.db_manager)
        self.tabs.addTab(self.db_viewer, "View Database")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def on_cancel_processing(self):
        """Cancel the current processing operation."""
        if self.current_processing and self.current_processing.isRunning():
            self.current_processing.cancel()
            self.progress_text.append("Cancelling operation...")
            self.cancel_btn.setEnabled(False)
            # Clear the queue to prevent further processing
            cancelled_count = len(self.queue)
            self.queue.clear()
            if cancelled_count > 0:
                self.progress_text.append(f"Removed {cancelled_count} items from queue")

    def should_skip_folder(self, folder_path):
        if not self.enable_folder_filter.isChecked():
            return False
        
        filter_text = self.folder_filter_input.text().strip()
        if not filter_text:
            return False
        
        try:
            excluded = [name.strip().lower() for name in filter_text.split(',') 
                       if name.strip()]
            if not excluded:
                return False
                
            folder_name = Path(folder_path).name.lower()
            return any(excluded_term in folder_name for excluded_term in excluded)
        except Exception as e:
            print(f"Error in folder filtering: {e}")
            return False

    def should_skip_sample_path(self, sample_path):
        """Check if a sample path should be skipped based on folder filter terms."""
        if not self.enable_folder_filter.isChecked():
            return False
        
        filter_text = self.folder_filter_input.text().strip()
        if not filter_text:
            return False
        
        try:
            excluded = [name.strip().lower() for name in filter_text.split(',') 
                       if name.strip()]
            if not excluded:
                return False
                
            # Check all path components, not just the filename
            path_parts = Path(sample_path).parts
            path_lower = sample_path.lower()
            
            return any(
                excluded_term in path_lower or 
                any(excluded_term in part.lower() for part in path_parts)
                for excluded_term in excluded
            )
        except Exception as e:
            print(f"Error in sample path filtering: {e}")
            return False

    def find_project_files_with_filtering(self, folder_path):
        project_files = []
        try:
            for root, dirs, files in os.walk(folder_path):
                # Filter directories if enabled
                if self.enable_folder_filter.isChecked():
                    dirs_to_remove = []
                    for d in list(dirs):
                        dir_path = os.path.join(root, d)
                        if self.should_skip_folder(dir_path):
                            dirs_to_remove.append(d)
                            self.progress_text.append(f"Skipping folder: {d}")
                    for d in dirs_to_remove:
                        if d in dirs:
                            dirs.remove(d)
                
                # Check for project bundles (directories with project extensions)
                for d in list(dirs):
                    if d.lower().endswith(PROJECT_EXTENSIONS):
                        project_path = os.path.join(root, d)
                        project_files.append(project_path)
                        # Don't recurse into project bundles
                        dirs.remove(d)
                
                # Check for individual project files (like .als files)
                for fn in files:
                    if fn.lower().endswith(PROJECT_EXTENSIONS):
                        project_files.append(os.path.join(root, fn))
                        
        except Exception as e:
            self.progress_text.append(f"Error scanning folder {folder_path}: {e}")
        return project_files

    def dragEnterEvent(self, event):
        if self.tabs.currentIndex() == 0 and event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if os.path.isdir(path) or path.lower().endswith(PROJECT_EXTENSIONS):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        if self.tabs.currentIndex() != 0:
            return
        urls = event.mimeData().urls()
        if not urls:
            return
        skipped_folders = []
        for url in urls:
            path = url.toLocalFile()
            
            # Check if this is a project file (either regular file or bundle directory)
            # Strip trailing slash for directory bundles
            clean_path = path.rstrip('/')
            is_project_file = clean_path.lower().endswith(PROJECT_EXTENSIONS)
            
            if is_project_file:
                # This is a project file (could be file or bundle directory)
                if os.path.isdir(path) and self.should_skip_folder(path):
                    skipped_folders.append(Path(path).name)
                    continue
                self.queue.append((path, True))
                file_type = "project bundle" if os.path.isdir(path) else "project"
                self.progress_text.append(f"Queued {file_type}: {Path(path).name}")
            elif os.path.isdir(path):
                # This is a regular folder - scan for project files inside
                if self.should_skip_folder(path):
                    skipped_folders.append(Path(path).name)
                    continue
                found = self.find_project_files_with_filtering(path)
                # convert to (path, True) to force processing (same as before)
                self.queue.extend([(fp, True) for fp in found])
                self.progress_text.append(f"Queued {len(found)} project files from {Path(path).name}")

        if skipped_folders:
            self.progress_text.append(f"Skipped {len(skipped_folders)} folders: {', '.join(skipped_folders)}")

        if self.queue and not self.current_processing:
            self.start_next_in_queue()
        elif not self.queue and skipped_folders:
            self.progress_text.append("All dropped folders were filtered out. No project files to process.")

    def on_scan_folder(self):
        """Open folder and enqueue projects with cleanup and force processing."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Scan")
        if not folder:
            return
        
        # Use the existing filtering logic
        project_files = self.find_project_files_with_filtering(folder)
        
        if not project_files:
            # Show more detailed message about what was filtered
            filter_status = " (with folder filtering applied)" if self.enable_folder_filter.isChecked() else ""
            QMessageBox.information(self, "No Projects", 
                                  f"No .als or .logicx projects found under:\n{folder}{filter_status}")
            return

        # Show what was found before processing
        file_list = "\n".join([f"  • {Path(p).name}" for p in project_files[:10]])  # Show first 10
        if len(project_files) > 10:
            file_list += f"\n  ... and {len(project_files) - 10} more"
        
        filter_note = "\n\n(Folder filtering is applied)" if self.enable_folder_filter.isChecked() else ""
        
        reply = QMessageBox.question(self, "Confirm Scan", 
                                    f"Found {len(project_files)} project files.{filter_note}\n"
                                    f"This will remove existing entries and re-scan. Continue?",
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply != QMessageBox.Yes:
            return

        # Convert to (path, force=True, cleanup=True) to ensure processing happens
        items = [(p, True, True) for p in project_files]
        self.enqueue_projects(items)
        self.progress_text.append(f"Queued {len(items)} projects for scan with cleanup.")

    def on_full_rescan(self):
        """Force re-scan of all projects known in the DB with cleanup, respecting folder filters."""
        projects = self.db_manager.get_all_projects()
        if not projects:
            QMessageBox.information(self, "No Projects", "No known projects in the database to re-scan.")
            return
        
        # Apply folder filtering to known projects
        if self.enable_folder_filter.isChecked():
            original_count = len(projects)
            filtered_projects = []
            for pfile, ptype in projects:
                if not self.should_skip_folder(Path(pfile).parent):
                    filtered_projects.append((pfile, ptype))
                else:
                    self.progress_text.append(f"Skipping filtered project: {Path(pfile).name}")
            projects = filtered_projects
            
            if len(projects) < original_count:
                self.progress_text.append(f"Folder filtering reduced projects from {original_count} to {len(projects)}")
        
        if not projects:
            QMessageBox.information(self, "No Projects", 
                                  "No projects remain after applying folder filters.")
            return
        
        filter_note = "\n(Folder filtering applied)" if self.enable_folder_filter.isChecked() else ""
        reply = QMessageBox.question(self, "Confirm Full Re-scan",
                                    f"Re-scan {len(projects)} known projects?{filter_note}\n"
                                    "This will remove existing entries and re-scan all. Continue?",
                                    QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
            
        items = [(pfile, True, True) for (pfile, ptype) in projects]  # force=True, cleanup=True
        self.enqueue_projects(items)
        self.progress_text.append(f"Queued {len(projects)} projects for full re-scan with cleanup.")

    def start_next_in_queue(self):
        if not self.queue:
            self.reset_ui()
            return
        next_item = self.queue.pop(0)
        if isinstance(next_item, tuple) and len(next_item) == 3:
            next_file, force_flag, cleanup_flag = next_item
        elif isinstance(next_item, tuple):
            next_file, force_flag = next_item
            cleanup_flag = False
        else:
            next_file, force_flag, cleanup_flag = next_item, True, False
            
        self.progress_text.append(f"Starting import: {Path(next_file).name} (force={force_flag}, cleanup={cleanup_flag})")
        self.current_processing = ProcessingThread(
            next_file,
            self.db_manager,
            update_comments=False,
            skip_folder_fn=self.should_skip_folder,
            skip_sample_fn=self.should_skip_sample_path,  # Add this line
            cleanup_existing=cleanup_flag
        )
        # pass the force flag to thread by attaching attribute and reading in run()
        self.current_processing.force = bool(force_flag)
        self.current_processing.progress_signal.connect(self.update_progress)
        self.current_processing.finished_signal.connect(self.queue_file_finished)
        self.current_processing.start()
        
        # Enable cancel button
        self.cancel_btn.setEnabled(True)

    def enqueue_projects(self, project_paths_with_force):
        """
        Accept list of (project_path, force_bool) or (project_path, force_bool, cleanup_bool) tuples 
        and queue them for processing. If nothing is currently processing, start the queue.
        """
        for item in project_paths_with_force:
            if len(item) == 3:
                p, f, c = item
                self.queue.append((p, bool(f), bool(c)))
                self.progress_text.append(f"Queued project: {Path(p).name} (force={f}, cleanup={c})")
            else:
                p, f = item
                self.queue.append((p, bool(f), False))
                self.progress_text.append(f"Queued project: {Path(p).name} (force={f})")
        
        if self.queue and not self.current_processing:
            self.start_next_in_queue()


    def queue_file_finished(self, message, total_files, updated_files):
        self.progress_text.append(f"{message}\nCompleted.\n")
        if hasattr(self, 'db_viewer'):
            self.db_viewer.refresh_data()
        # Continue with next queued file
        QTimer.singleShot(50, self.start_next_in_queue)

    def update_progress(self, msg):
        self.progress_text.append(msg)
        sb = self.progress_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def reset_ui(self):
        self.drop_label.setText("Drag & drop a project file or folder here")
        self.cancel_btn.setEnabled(False)  # Add this line
        if self.current_processing:
            try:
                self.current_processing.quit()
                self.current_processing.wait()
            except Exception:
                pass
            self.current_processing = None

# ---------------------------
# App entrypoint
# ---------------------------

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
