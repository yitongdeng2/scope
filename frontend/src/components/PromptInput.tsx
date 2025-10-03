import { useState } from "react";
import { Input } from "./ui/input";
import { Button } from "./ui/button";
import { ArrowUp } from "lucide-react";

interface PromptInputProps {
  className?: string;
  currentPrompt: string;
  onPromptChange?: (prompt: string) => void;
  onPromptSubmit?: (prompt: string) => void;
  disabled?: boolean;
}

export function PromptInput({
  className = "",
  currentPrompt,
  onPromptChange,
  onPromptSubmit,
  disabled = false,
}: PromptInputProps) {
  const [isProcessing, setIsProcessing] = useState(false);

  const handleSubmit = () => {
    if (!currentPrompt.trim()) return;

    setIsProcessing(true);

    // Send the prompt update via data channel
    if (onPromptSubmit) {
      onPromptSubmit(currentPrompt.trim());
    }

    // Reset processing state after a short delay
    setTimeout(() => {
      setIsProcessing(false);
    }, 1000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSubmit();
    }
  };

  return (
    <div
      className={`flex items-center bg-card border border-border rounded-full px-4 py-3 gap-3 ${className}`}
    >
      <Input
        placeholder="blooming flowers"
        value={currentPrompt}
        onChange={e => onPromptChange?.(e.target.value)}
        onKeyPress={handleKeyPress}
        disabled={disabled}
        className="flex-1 bg-transparent border-0 text-card-foreground placeholder:text-muted-foreground focus-visible:ring-0 focus-visible:ring-offset-0 p-0 disabled:opacity-50 disabled:cursor-not-allowed"
      />
      <Button
        onClick={handleSubmit}
        disabled={disabled || !currentPrompt.trim() || isProcessing}
        size="sm"
        className="rounded-full w-8 h-8 p-0 bg-black hover:bg-gray-800 text-white disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isProcessing ? "..." : <ArrowUp className="h-4 w-4" />}
      </Button>
    </div>
  );
}
