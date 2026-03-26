import boto3
import os
import uuid
from PIL import Image
import io
import logging
from dotenv import load_dotenv
from urllib.parse import urlparse, unquote

load_dotenv()
logger = logging.getLogger(__name__)


AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME","uploads")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

def upload_file_to_s3(file_bytes: bytes, filename: str, folder: str, content_type: str):
    key = f"{folder}/{uuid.uuid4()}-{filename}"
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=file_bytes,
        ContentType=content_type,
        # ACL="public-read"   # IMPORTANT
    )
    url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"
    return url


def get_s3_key_from_url(file_url: str) -> str | None:
    if not file_url:
        return None
    try:
        parsed = urlparse(file_url)
        # remove leading /
        key = parsed.path.lstrip("/")
        # decode URL encoded characters
        key = unquote(key)
        logger.info(f"Key: {key}")
        return key
    except Exception:
        return None


def delete_file_from_s3(file_url: str):
    key = get_s3_key_from_url(file_url)
    if not key:
        return
    try:
        s3.delete_object(Bucket=BUCKET_NAME, Key=key)
        logger.debug(f"S3 deleted: {key}")
    except Exception as e:
        logger.error(f"Failed to delete the image from S3 bucket")

def load_image_from_s3(image_url: str):
    key = get_s3_key_from_url(image_url)
    if not key:
        return
    try:
        response = s3.get_object(Bucket=BUCKET_NAME,Key=key)
        body = response.get("Body")
        if body is None:
            raise ValueError("S3 bucket returned empty body")
        logger.info(f"S3 bucket body: {body}")
        image_bytes = body.read()
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        logger.error(f"AWS error {e}")

