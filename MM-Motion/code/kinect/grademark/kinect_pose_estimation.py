import cv2
import mediapipe as mp
import os

# -------------------------- 核心配置（需根据你的情况修改） --------------------------
# 1. 输入视频路径（原Kinect视频，无中文/空格）
INPUT_VIDEO_PATH = "D:/visualization-project/fullBody/fullBody_split.mp4"
# 2. 输出视频路径（保存处理后的视频，建议和输入同目录）
OUTPUT_VIDEO_PATH = "D:/visualization-project/fullBody/fullBody_split_with_pose.mp4"

def main():
    # 初始化mediapipe姿态检测器
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
        smooth_landmarks=True  # 平滑关节点，减少抖动
    ) as pose:

        # -------------------------- 1. 打开输入视频并获取参数 --------------------------
        # 强制用FFMPEG解码器打开，避免格式兼容性问题
        cap = cv2.VideoCapture(INPUT_VIDEO_PATH, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"❌ 无法打开输入视频：{INPUT_VIDEO_PATH}")
            return

        # 获取原视频的关键参数（保持输出视频和原视频一致）
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 宽度（像素）
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高度（像素）
        fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率（FPS）
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数（用于显示进度）
        print(f"✅ 输入视频信息：{frame_width}x{frame_height} | {fps:.1f} FPS | 共{total_frames}帧")


        # -------------------------- 2. 初始化视频写入器（保存输出视频） --------------------------
        # 视频编码格式：MP4推荐使用 cv2.VideoWriter_fourcc(*'mp4v')（兼容性强）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # 创建视频写入对象（参数：输出路径、编码格式、帧率、分辨率）
        out = cv2.VideoWriter(
            OUTPUT_VIDEO_PATH,
            fourcc,
            fps,
            (frame_width, frame_height)  # 必须和输入视频分辨率一致
        )

        if not out.isOpened():
            print(f"❌ 无法创建输出视频：{OUTPUT_VIDEO_PATH}")
            cap.release()  # 释放输入视频资源
            return


        # -------------------------- 3. 处理视频帧并写入结果 --------------------------
        print("✅ 开始处理视频（按 'q' 可提前退出）")
        processed_frames = 0  # 已处理帧数（用于显示进度）

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # 所有帧处理完毕，退出循环

            # 进度提示（每处理10帧显示一次，避免刷屏）
            processed_frames += 1
            if processed_frames % 10 == 0:
                progress = (processed_frames / total_frames) * 100
                print(f"🔄 处理进度：{processed_frames}/{total_frames}帧（{progress:.1f}%）")

            # -------------------------- 姿态检测与绘制 --------------------------
            # 转换为RGB格式（mediapipe仅支持RGB输入）
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False  # 禁用写入，提升处理速度

            # 核心：检测人体姿态（获取33个关节点）
            results = pose.process(rgb_frame)

            # 转回BGR格式（OpenCV显示/写入需BGR）
            rgb_frame.flags.writeable = True
            output_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # 绘制关节点和骨骼（仅当检测到人体时绘制）
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    output_frame,  # 要绘制的帧
                    results.pose_landmarks,  # 关节点数据
                    mp_pose.POSE_CONNECTIONS,  # 骨骼连接关系
                    # 关节点样式：绿色实心圆（厚度3，半径4，醒目且不遮挡细节）
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 255, 0), thickness=3, circle_radius=4
                    ),
                    # 骨骼样式：紫色线条（厚度2，清晰区分骨骼）
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(121, 44, 250), thickness=2
                    )
                )


            # -------------------------- 4. 写入处理后的帧到输出视频 --------------------------
            out.write(output_frame)

            # 显示实时处理效果（按 'q' 可提前退出）
            cv2.namedWindow("Kinect姿态检测（实时预览）", cv2.WINDOW_NORMAL)
            cv2.imshow("Kinect姿态检测（实时预览）", output_frame)
            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                print("🔚 用户手动提前退出")
                break


        # -------------------------- 5. 释放资源（关键！避免文件损坏） --------------------------
        # 必须先释放写入器，再释放输入视频，否则输出视频可能损坏
        out.release()
        cap.release()
        cv2.destroyAllWindows()

        # 处理完成提示
        if processed_frames == total_frames:
            print(f"\n✅ 视频处理完成！")
            print(f"📁 输出视频路径：{OUTPUT_VIDEO_PATH}")
            print(f"📊 处理统计：共{total_frames}帧，输出分辨率{frame_width}x{frame_height}，帧率{fps:.1f} FPS")
        else:
            print(f"\n⚠️  视频未完全处理（提前退出）")
            print(f"📁 已生成部分视频：{OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()