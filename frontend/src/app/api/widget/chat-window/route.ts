import { NextResponse } from 'next/server';
import path from 'path';
import { promises as fs } from 'fs';

export async function GET() {
  try {
    // Path to the chat-window.html file in the public directory
    const filePath = path.join(process.cwd(), 'public', 'widget', 'chat-window.html');
    const fileContent = await fs.readFile(filePath, 'utf8');
    
    return new NextResponse(fileContent, {
      headers: {
        'Content-Type': 'text/html',
        // Add CORS headers to allow embedding in other domains
        'Access-Control-Allow-Origin': '*',
      },
    });
  } catch (error) {
    console.error('Error loading chat window:', error);
    return new NextResponse('Error loading chat window', { status: 500 });
  }
}

// Handle OPTIONS method for CORS preflight
// This is required for cross-origin requests in some browsers
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}
