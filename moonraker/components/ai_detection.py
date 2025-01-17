from __future__ import annotations
import logging
import time
import os
import asyncio
import aiohttp
from typing import TYPE_CHECKING, Dict, Any, Optional

if TYPE_CHECKING:
    from ..confighelper import ConfigHelper, AIConfig
    from ..server import Server
    from ..common import WebRequest
    from .webcam import WebcamManager
    from .klippy_connection import KlippyConnection as Klippy

class AiDetection:
    def __init__(self, config: ConfigHelper) -> None:
        self.server = config.get_server()
        app = self.server.lookup_component("application")
        self.ai_config: Optional[AIConfig] = app.ai_config
        if self.ai_config is None:
            try:
                ai_cfg = config.getsection('ai')
                if ai_cfg is not None:
                    self.ai_config = AIConfig(ai_cfg)
            except config.error:
                raise config.error(
                    "[ai] section not found in configuration"
                )
        
        # 注册API端点
        self.server.register_endpoint(
            "/api/v1/device/print/image", ['POST'], 
            self._handle_image_upload,
            auth_required=True
        )
        self.server.register_endpoint(
            "/api/v1/predict", ['POST'],
            self._handle_predict,
            auth_required=True
        )

        # 注册G-code命令处理器
        self.server.register_event_handler(
            "server:klippy_ready", self._handle_klippy_ready
        )
        self.server.register_event_handler(
            "server:klippy_shutdown", self._handle_klippy_shutdown
        )

    async def _handle_klippy_ready(self) -> None:
        """当Klippy准备就绪时注册命令"""
        klippy: Klippy = self.server.lookup_component("klippy_connection")
        # 注册action:ai_snapshot命令处理
        await klippy.request(
            WebRequest("gcode/subscribe_output", 
                      {"response_template": "action:ai_snapshot"})
        )

    async def _handle_klippy_shutdown(self) -> None:
        """Klippy关闭时的处理"""
        pass

    async def _handle_gcode_response(self, response: str) -> None:
        """处理G-code响应"""
        if response.startswith("action:ai_snapshot"):
            await self._handle_snapshot_request()

    async def _request_prediction(self, image_path: str, task_id: str) -> None:
        """请求AI预测
        
        参数:
            image_path: 图片路径
            task_id: 任务ID
        """
        try:
            # 构建图片URL
            host_info = self.server.get_host_info()
            image_name = os.path.basename(image_path)
            image_url = (
                f"http://{host_info['address']}/server/files/ai_snapshots/{image_name}"
            )
            
            # 构建回调URL
            callback_url = f"http://{host_info['address']}/api/v1/ai/callback"
            
            # 准备请求数据
            predict_data = {
                "image_url": image_url,
                "task_id": task_id,
                "callback_url": callback_url
            }
            
            # 发送预测请求
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ai_config.base_url}/predict",
                    json=predict_data,
                    timeout=self.ai_config.upload_timeout
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise self.server.error(
                            f"AI service returned error: {error_text}",
                            resp.status
                        )
                    result = await resp.json()
                    logging.info(
                        f"AI Detection: Prediction requested for task {task_id}, "
                        f"response: {result}"
                    )
                    return result
                    
        except asyncio.TimeoutError:
            logging.error(
                f"AI Detection: Prediction request timeout for task {task_id}"
            )
            raise self.server.error(
                "AI service request timeout", 
                504
            )
        except aiohttp.ClientError as e:
            logging.error(
                f"AI Detection: Failed to connect to AI service for task {task_id}: {e}"
            )
            raise self.server.error(
                f"Failed to connect to AI service: {str(e)}", 
                503
            )

    async def _handle_predict(self, web_request: WebRequest) -> Dict[str, Any]:
        """处理预测请求
        
        参数:
            web_request: 包含预测请求参数的WebRequest对象
            
        返回:
            Dict包含处理结果
        """
        request_json = web_request.get_json()
        
        # 验证必需参数
        required_fields = ['image_url', 'task_id', 'callback_url']
        for field in required_fields:
            if field not in request_json:
                raise self.server.error(
                    f"Missing required field: {field}", 
                    400
                )
        
        try:
            # 发送预测请求到AI服务
            result = await self._request_prediction(
                request_json['image_url'],
                request_json['task_id']
            )
            return {
                'code': 200,
                'message': 'success',
                'data': result
            }
        except Exception as e:
            logging.exception("Error processing prediction request")
            raise self.server.error(str(e), 500)

    async def _handle_snapshot_request(self) -> None:
        """处理拍照请求"""
        try:
            # 获取WebcamManager组件
            webcam: WebcamManager = self.server.lookup_component("webcam")
            
            # 生成任务ID
            task_id = f"snapshot_{int(time.time())}"
            
            # 拍照
            snapshot = await webcam.take_snapshot()
            if snapshot is None:
                logging.error("Failed to take snapshot")
                return
                
            # 生成保存路径
            save_filename = f"{task_id}.jpg"
            save_path = os.path.join(self.ai_config.snapshot_path, save_filename)
            
            # 保存图片
            with open(save_path, 'wb') as f:
                f.write(snapshot)
                
            logging.info(
                f"AI Detection: Snapshot saved for task {task_id} "
                f"at {save_path}"
            )
            
            # 添加预测请求
            await self._request_prediction(save_path, task_id)
            
        except Exception as e:
            logging.exception("Error processing snapshot request")
            raise self.server.error(
                f"Failed to process snapshot: {str(e)}", 500
            )

    async def _handle_image_upload(self, web_request: WebRequest) -> Dict[str, Any]:
        """处理图片上传请求
        
        参数:
            web_request: WebRequest对象,包含上传的文件和参数
            
        返回:
            Dict包含处理结果
            
        错误:
            400: 参数错误
            500: 服务器内部错误
        """
        # 验证请求参数
        if 'file' not in web_request.files:
            raise self.server.error("No file uploaded", 400)
        
        file = web_request.files['file']
        if not file or not file.filename:
            raise self.server.error("Empty file", 400)
            
        task_id = web_request.get_str('task_id')
        if not task_id:
            raise self.server.error("task_id is required", 400)
        
        # 验证文件类型
        filename = file.filename.lower()
        if not filename.endswith(('.jpg', '.jpeg', '.png')):
            raise self.server.error(
                "Invalid file type. Only jpg/jpeg/png allowed", 400
            )
        
        # 生成文件名和保存路径
        timestamp = int(time.time())
        save_filename = f"{task_id}_{timestamp}{os.path.splitext(filename)[1]}"
        save_path = os.path.join(self.ai_config.snapshot_path, save_filename)
        
        # 保存文件
        try:
            with open(save_path, 'wb') as f:
                f.write(file.body)
            logging.info(
                f"AI Detection: Image saved for task {task_id} "
                f"at {save_path}"
            )
        except Exception as e:
            logging.exception("Failed to save uploaded image")
            raise self.server.error(
                f"Failed to save image: {str(e)}", 500
            )
            
        return {
            'code': 200,
            'message': 'success',
            'data': None
        }

def load_component(config: ConfigHelper) -> AiDetection:
    return AiDetection(config) 