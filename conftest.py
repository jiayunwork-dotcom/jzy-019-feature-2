import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

# 每个测试会话用独立的临时 SQLite，避免残留数据导致重名冲突；
# 必须在导入 app.main（模块级建库）之前设置。
_db_fd, _db_path = tempfile.mkstemp(prefix="optics_test_", suffix=".db")
os.close(_db_fd)
os.unlink(_db_path)  # 让 SQLite 自己建库，避免空文件被当成坏库
os.environ["OPTICS_DB"] = _db_path
