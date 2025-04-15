"use client";

import { useRef, useState, useEffect } from "react";

export default function DrawCanvas({
	onFinish,
}: {
	onFinish: (data: number[][]) => void;
}) {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const [drawing, setDrawing] = useState(false);
	const [image, setImage] = useState<number[][]>([[]]);

	useEffect(() => {
		const canvas = canvasRef.current as HTMLCanvasElement;
		const ctx = canvas?.getContext("2d");

		if (ctx) {
			ctx.fillStyle = "black";
			ctx.fillRect(0, 0, canvas!.width, canvas!.height);
		}
	}, []);

	const getPos = (e: React.MouseEvent) => {
		const rect = canvasRef.current!.getBoundingClientRect();

		return {
			x: e.clientX - rect.left,
			y: e.clientY - rect.top,
		};
	};

	const startDraw = (e: React.MouseEvent) => {
		setDrawing(true);
		draw(e);
	};

	const stopDraw = () => {
		setDrawing(false);

		// downscale for model eval
		const smallCanvas = document.createElement("canvas");
		smallCanvas.width = 28;
		smallCanvas.height = 28;

		const ctx = smallCanvas.getContext("2d")!;
		ctx.drawImage(canvasRef.current!, 0, 0, 28, 28);

		const imageData = ctx.getImageData(0, 0, 28, 28);
		const pixels = imageData.data;

		const grayscale: number[][] = [];

		for (let i = 0; i < 28; i++) {
			grayscale.push([]);

			for (let j = 0; j < 28; j++) {
				const index = (i * 28 + j) * 4;
				const r = pixels[index];
				const val = r / 255;

				grayscale[i].push(val);
			}
		}

		setImage(grayscale);
	};

	const draw = (e: React.MouseEvent) => {
		if (!drawing) return;

		const ctx = canvasRef.current!.getContext("2d")!;
		const { x, y } = getPos(e);

		ctx.fillStyle = "white";
		ctx.beginPath();
		ctx.arc(x, y, 10, 0, 2 * Math.PI);
		ctx.fill();
	};

	const clearCanvas = () => {
		const canvas = canvasRef.current;
		const ctx = canvas?.getContext("2d");

		if (ctx) {
			ctx.fillStyle = "black";
			ctx.fillRect(0, 0, canvas!.width, canvas!.height);
		}
	};

	return (
		<section className="flex flex-col gap-2">
			<div>
				<canvas
					ref={canvasRef}
					width={280}
					height={280}
					onMouseDown={startDraw}
					onMouseMove={draw}
					onMouseUp={stopDraw}
					style={{
						border: "2px solid #444",
						background: "black",
						touchAction: "none",
						borderRadius: "20px",
					}}
				/>
			</div>

			<div className="pt-2 flex justify-end gap-2">
				<button
					onClick={clearCanvas}
					className="bg-gradient-to-r from-neutral-100 to-neutral-300 text-black border border-white rounded-full px-4 py-1 font-medium text-lg"
				>
					clear
				</button>

				<button
					onClick={() => onFinish(image)}
					className="bg-gradient-to-r from-neutral-100 to-neutral-300 text-black rounded-full px-4 py-1 font-medium text-lg"
				>
					guess
				</button>
			</div>
		</section>
	);
}
