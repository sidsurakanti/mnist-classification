"use client";

import DrawCanvas from "@/components/DrawingCanvas";
import { useState } from "react";
import { mono } from "@/styles/fonts";

export default function Home() {
	const [prediction, setPrediction] = useState<number>();

	const handleFinish = async (image: number[][]) => {
		const res = await fetch("http://localhost:8000/predict", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ data: image }),
		});
		const prediction = await res.json();
		setPrediction(Number(prediction.prediction));
	};

	return (
		<main className="w-full h-full">
			<section className="h-9/10 flex flex-col gap-20 items-center justify-center">
				<DrawCanvas onFinish={handleFinish} />
				{/* <div className="text-5xl">{prediction}</div> */}

				<div className="flex gap-2">
					{Array.from({ length: 10 }, (_, i) => i).map((n) => (
						<span
							key={n}
							className={`transition-all h-24 w-24 p-6 text-[42px] font-bold text-neutral-800 rounded-xl flex justify-center items-center border-2 ${
								mono.className
							} ${
								prediction == n
									? "bg-emerald-200 border-emerald-300/70 -translate-y-2 shadow-lg"
									: "bg-white border-stone-200/60 shadow-sm"
							} `}
						>
							{n}
						</span>
					))}
				</div>
			</section>
			<footer
				className={`${mono.className} h-1/10 flex flex-col justify-end items-end p-4`}
			>
				<p>MNIST Digit Classification</p>
				<p>Model: Pytorch NN (269,322 parameters)</p>
				<a
					href="https://github.com/sidsurakanti/mnist-classification"
					target="_blank"
					className="text-blue-600 font-medium hover:underline underline-offset-4"
				>
					Source code
				</a>
				Sid, 2025
			</footer>
		</main>
	);
}
