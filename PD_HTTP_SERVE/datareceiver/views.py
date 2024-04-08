from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
# Create your views here.
from datareceiver.multimaps import MyHttp,save_data_info
from datareceiver.loader_detect import pd_detect

@csrf_exempt  # 仅用于测试，实际部署时请务必使用CSRF保护
@require_http_methods(["POST"])  # 限制只允许POST请求
def receive_data(request):
    # print("文件内容是{}".format(request.FILES))
    if request.method == "POST":
        try:
            # 解析 JSON 数据
            data_info = json.loads(request.POST.get('data_info'))
            # 获取二进制文件
            file_data = request.FILES['file'].read()

            device_type = data_info['device_type']
            # protocol_ver = data_info['protocol_ver']
            detection_type = data_info['detection_type']
            file_name = data_info['file_name']
            file_time = data_info['file_time']
            try:
                Ins = MyHttp()
                Ins.process_complete_data(file_data, data_info)
                su = {"sucess": "数据保存成功", "file_name":file_name,"status_code": 1}
                print(su)
                # return JsonResponse(ret_msg, status=200)
            except Exception as e:
                ret_msg = {"error": f"数据保存失败: {e}", "status_code": 301}
                return JsonResponse(ret_msg, status=301)

            if detection_type == "AE"or detection_type == "AD":
                try:
                    # AI识别
                    counts, predicted_labels, predicted_prob = pd_detect(file_name)
                    ret_msg = {"status_code": 1,
                        "discharge_time": file_time,
                            "discharge_counts": counts,
                            "discharge_type": predicted_labels,
                            "probability": predicted_prob}
                    return JsonResponse(ret_msg, status=200)
                except Exception as e:
                    ret_msg = {"error": f"AI识别失败: {e}", "status_code": 302}
                    return JsonResponse(ret_msg, status=302)
            else:
                ret_msg = {"status_code": 1,
                    "discharge_time": file_time,
                    "discharge_counts": 0,
                    "discharge_type": 1,
                    "probability": 0.0}
                return JsonResponse(ret_msg, status=200)
                
        except Exception as e:
            # 其他错误
            ret_msg = {"error": f"服务器内部错误: {e}", "status_code": 500}
            status_code = 500
            return JsonResponse(ret_msg, status=500)
