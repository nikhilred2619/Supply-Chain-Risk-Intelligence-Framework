from pathlib import Path
import base64, tarfile, io

payload = Path("cloud_runner_payload.b64").read_text().strip()
data = base64.b64decode(payload)
with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
    tf.extractall(".")
print("Cloud runner payload extracted.")
