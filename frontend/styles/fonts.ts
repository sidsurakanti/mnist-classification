import { Poppins } from "next/font/google";
import { Fira_Code } from "next/font/google";

export const poppins = Poppins({
	subsets: ["latin"],
	weight: ["400", "500", "600", "700", "800"],
});

export const mono = Fira_Code({
	subsets: ["latin"],
	weight: ["400", "500", "600", "700"],
});
