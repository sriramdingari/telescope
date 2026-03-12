import { useState } from "react";

export default function App() {
    const [count, setCount] = useState(0);
    return count + setCount.length;
}

export function renderApp() {
    return App();
}
