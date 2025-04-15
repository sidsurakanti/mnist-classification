"use client";

import DrawCanvas from "@/components/DrawingCanvas";
import { useState } from "react";

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
		<main className="w-full h-full flex justify-center items-center">
			<section className="flex flex-col gap-20 items-center">
				<DrawCanvas onFinish={handleFinish} />
				{/* <div className="text-5xl">{prediction}</div> */}

				<div className="flex gap-2">
					{Array.from({ length: 10 }, (_, i) => i).map((n) => (
						
							<span
								key={n}
								className={`h-24 w-24 p-6 text-4xl rounded-xl flex justify-center items-center border-2 border-neutral-500 ${
									prediction == n ? "bg-green-500 border-green-900" : "bg-neutral-950"
								} `}
							>
								{n}
							</span>
					))}
				</div>
			</section>
		</main>
	);
}
