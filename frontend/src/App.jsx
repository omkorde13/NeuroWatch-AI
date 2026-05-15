// frontend/src/App.jsx

import { useEffect, useState } from "react"

export default function NeuroWatchDashboard() {

  const [status, setStatus] = useState("LOADING...")
  const [threat, setThreat] = useState("LOW")
  const [fps, setFps] = useState(0)
  const [detections, setDetections] = useState(0)
  const [accuracy, setAccuracy] = useState(0)

  // ==========================
  // Backend + Webcam Setup
  // ==========================

  useEffect(() => {

    // ==========================
    // Fetch AI Status
    // ==========================

    const fetchStatus = async () => {

      try {

        const response = await fetch(
          "http://127.0.0.1:8000/status"
        )

        const data = await response.json()

        setStatus(data.status)

        setThreat(data.threat)

        setFps(data.fps)

        setDetections(data.detections)

        setAccuracy(data.accuracy)

      } catch (error) {

        console.error(error)

      }
    }

    fetchStatus()

    const interval = setInterval(
      fetchStatus,
      2000
    )

    // ==========================
    // Webcam Setup
    // ==========================

    const setupCamera = async () => {

      try {

        const stream =
          await navigator.mediaDevices.getUserMedia({
            video: true
          })

        const video =
          document.getElementById("webcam")

        if (video) {

          video.srcObject = stream

        }

      } catch (error) {

        console.error(
          "Camera Error:",
          error
        )

      }
    }

    setupCamera()

    return () => clearInterval(interval)

  }, [])

  return (

    <div className="min-h-screen bg-black text-white flex flex-col items-center justify-center p-6">

      <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Left Panel */}
        <div className="lg:col-span-2 bg-zinc-900 rounded-3xl shadow-2xl overflow-hidden border border-zinc-800">

          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">

            <div>

              <h1 className="text-3xl font-bold tracking-wide text-cyan-400">
                NeuroWatch AI
              </h1>

              <p className="text-zinc-400 text-sm mt-1">
                Real-Time Neuromorphic Surveillance System
              </p>

            </div>

            <div className="flex items-center gap-3">

              <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse"></div>

              <span className="text-green-400 font-medium">
                System Active
              </span>

            </div>

          </div>

          {/* Webcam Feed */}
          <div className="relative aspect-video bg-zinc-950 flex items-center justify-center">

            <video
              id="webcam"
              autoPlay
              muted
              playsInline
              className="w-full h-full object-cover"
            />

            {/* AI Status */}
            <div className="absolute top-4 left-4 bg-black/70 px-4 py-2 rounded-xl border border-zinc-700">

              <p className={`font-bold text-lg ${
                status === "VIOLENT ACTIVITY DETECTED"
                  ? "text-red-500"
                  : status === "SUSPICIOUS MOVEMENT"
                  ? "text-yellow-400"
                  : "text-green-400"
              }`}>
                {status}
              </p>

            </div>

            {/* AI Engine */}
            <div className="absolute bottom-4 right-4 bg-black/70 px-4 py-2 rounded-xl border border-zinc-700">

              <p className="text-cyan-300 text-sm">
                SNN Inference Running...
              </p>

            </div>

          </div>

        </div>

        {/* Right Panel */}
        <div className="bg-zinc-900 rounded-3xl shadow-2xl border border-zinc-800 p-6 flex flex-col gap-6">

          {/* Analytics */}
          <div>

            <h2 className="text-2xl font-bold text-cyan-400 mb-2">
              Live Analytics
            </h2>

            <p className="text-zinc-400 text-sm">
              Real-time spike-based surveillance monitoring.
            </p>

          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 gap-4">

            {/* Accuracy */}
            <div className="bg-zinc-950 rounded-2xl p-4 border border-zinc-800">

              <p className="text-zinc-400 text-sm">
                Model Accuracy
              </p>

              <h3 className="text-3xl font-bold text-green-400 mt-2">
                {accuracy}%
              </h3>

            </div>

            {/* FPS */}
            <div className="bg-zinc-950 rounded-2xl p-4 border border-zinc-800">

              <p className="text-zinc-400 text-sm">
                FPS
              </p>

              <h3 className="text-3xl font-bold text-cyan-400 mt-2">
                {fps}
              </h3>

            </div>

            {/* Detections */}
            <div className="bg-zinc-950 rounded-2xl p-4 border border-zinc-800">

              <p className="text-zinc-400 text-sm">
                Detections
              </p>

              <h3 className="text-3xl font-bold text-yellow-400 mt-2">
                {detections}
              </h3>

            </div>

            {/* Threat */}
            <div className="bg-zinc-950 rounded-2xl p-4 border border-zinc-800">

              <p className="text-zinc-400 text-sm">
                Threat Level
              </p>

              <h3 className={`text-3xl font-bold mt-2 ${
                threat === "HIGH"
                  ? "text-red-500"
                  : threat === "MEDIUM"
                  ? "text-yellow-400"
                  : "text-green-400"
              }`}>
                {threat}
              </h3>

            </div>

          </div>

          {/* Logs */}
          <div className="bg-zinc-950 rounded-2xl p-4 border border-zinc-800 flex-1 overflow-hidden">

            <div className="flex items-center justify-between mb-4">

              <h3 className="text-lg font-bold text-white">
                Activity Logs
              </h3>

              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>

            </div>

            <div className="space-y-3 text-sm overflow-y-auto max-h-[300px] pr-2">

              <div className="bg-zinc-900 rounded-xl p-3 border border-zinc-800">

                <p className="text-green-400 font-semibold">
                  Neural Spike Stream Active
                </p>

                <p className="text-zinc-500 text-xs mt-1">
                  Live SNN inference initialized
                </p>

              </div>

              <div className="bg-zinc-900 rounded-xl p-3 border border-zinc-800">

                <p className="text-cyan-400 font-semibold">
                  Optical Flow Monitoring
                </p>

                <p className="text-zinc-500 text-xs mt-1">
                  Temporal motion tracking enabled
                </p>

              </div>

              <div className="bg-zinc-900 rounded-xl p-3 border border-zinc-800">

                <p className="text-yellow-400 font-semibold">
                  AI Surveillance Running
                </p>

                <p className="text-zinc-500 text-xs mt-1">
                  Real-time violence classification active
                </p>

              </div>

            </div>

          </div>

        </div>

      </div>

      {/* Footer */}
      <div className="mt-8 text-zinc-500 text-sm text-center">

        NeuroWatch AI • Spiking Neural Network Surveillance Platform

      </div>

    </div>
  )
}