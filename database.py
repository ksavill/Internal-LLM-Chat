import sqlite3
from datetime import datetime
import json
import uuid
import os
import hashlib

def get_db_connection():
    conn = sqlite3.connect('chats.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    
    # Create the users table if it doesn't exist
    conn.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password_hash TEXT,
        salt TEXT
    )
    ''')
    
    # Create the conversations table if it doesn't exist.
    # Note: If the table already exists from an earlier version, the column won't be added automatically.
    conn.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        conversation_id TEXT PRIMARY KEY,
        user_id INTEGER,
        preview TEXT,
        created_at DATETIME,
        updated_at DATETIME,
        messages_json TEXT,
        backup_used INTEGER DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    ''')
    
    # Check if the backup_used column exists; if not, add it.
    cursor = conn.execute("PRAGMA table_info(conversations)")
    columns = [row["name"] for row in cursor.fetchall()]
    if "backup_used" not in columns:
        conn.execute("ALTER TABLE conversations ADD COLUMN backup_used INTEGER DEFAULT 0")

    if "openai_response_id" not in columns:
         conn.execute("ALTER TABLE conversations ADD COLUMN openai_response_id TEXT DEFAULT NULL")
    
    conn.commit()
    conn.close()

def hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16)
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return pwd_hash, salt

def create_user(username, password):
    conn = get_db_connection()
    cursor = conn.execute('SELECT * FROM users WHERE username = ?', (username,))
    if cursor.fetchone():
        conn.close()
        raise ValueError("Username already exists")
    pwd_hash, salt = hash_password(password)
    conn.execute('''
    INSERT INTO users (username, password_hash, salt)
    VALUES (?, ?, ?)
    ''', (username, pwd_hash.hex(), salt.hex()))
    conn.commit()
    conn.close()

def verify_user(username, password):
    conn = get_db_connection()
    cursor = conn.execute('SELECT user_id, password_hash, salt FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    if not user:
        return None
    stored_hash = bytes.fromhex(user['password_hash'])
    salt = bytes.fromhex(user['salt'])
    pwd_hash, _ = hash_password(password, salt)
    if pwd_hash == stored_hash:
        return user['user_id']
    return None

def create_conversation(user_id, messages):
    conn = get_db_connection()
    conversation_id = str(uuid.uuid4())
    preview = get_preview(messages)
    created_at = datetime.now()
    updated_at = created_at
    messages_json = json.dumps([m.dict() for m in messages])
    conn.execute('''
    INSERT INTO conversations (conversation_id, user_id, preview, created_at, updated_at, messages_json)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (conversation_id, user_id, preview, created_at, updated_at, messages_json))
    conn.commit()
    conn.close()
    return conversation_id

def update_conversation(conversation_id, messages, openai_response_id=None):
    conn = get_db_connection()
    updated_at = datetime.now()
    messages_json = json.dumps([m.dict() for m in messages])
    
    # If openai_response_id is provided, update that field as well
    if openai_response_id:
        conn.execute("""
        UPDATE conversations
        SET messages_json = ?, updated_at = ?, openai_response_id = ?
        WHERE conversation_id = ?
        """, (messages_json, updated_at, openai_response_id, conversation_id))
    else:
        conn.execute("""
        UPDATE conversations
        SET messages_json = ?, updated_at = ?
        WHERE conversation_id = ?
        """, (messages_json, updated_at, conversation_id))
    
    conn.commit()
    conn.close()

def get_user_conversations(user_id):
    conn = get_db_connection()
    cursor = conn.execute('''
    SELECT conversation_id, preview, created_at, updated_at
    FROM conversations
    WHERE user_id = ?
    ORDER BY updated_at DESC
    ''', (user_id,))
    conversations = cursor.fetchall()
    conn.close()
    return [dict(row) for row in conversations]

def get_conversation(conversation_id):
    conn = get_db_connection()
    cursor = conn.execute('''
    SELECT messages_json
    FROM conversations
    WHERE conversation_id = ?
    ''', (conversation_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row['messages_json'])
    return None

def get_preview(messages):
    for m in messages:
        if m.role == "user":
            return m.content[:50]
    return ""

def get_conversation_openai_response_id(conversation_id):
    """
    Retrieve the OpenAI response ID for a specific conversation.
    Used when sending follow-up requests to the OpenAI Responses API.
    """
    conn = get_db_connection()
    cursor = conn.execute('''
    SELECT openai_response_id
    FROM conversations
    WHERE conversation_id = ?
    ''', (conversation_id,))
    row = cursor.fetchone()
    conn.close()
    return row['openai_response_id'] if row and row['openai_response_id'] else None