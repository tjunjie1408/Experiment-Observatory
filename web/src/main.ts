import { mount } from "svelte";
import App from "./App.svelte";
import "./styles/tokens.css";
import "./styles/global.css";

const target = document.querySelector<HTMLDivElement>("#app");
if (target === null) throw new Error("missing #app root element");

mount(App, { target });
