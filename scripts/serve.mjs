import http from 'node:http';
import { createReadStream, existsSync, statSync } from 'node:fs';
import path from 'node:path';

const directory = path.resolve('dist');
const port = Number(process.env.PORT || 4173);
const types = { '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.json': 'application/json', '.xml': 'application/xml', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp', '.avif': 'image/avif', '.gif': 'image/gif', '.svg': 'image/svg+xml', '.ttf': 'font/ttf', '.woff2': 'font/woff2', '.woff': 'font/woff' };
http.createServer((request, response) => {
  let pathname;
  try { pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname); }
  catch { response.writeHead(400).end('Bad request'); return; }
  let file = path.resolve(directory, `.${pathname}`);
  if (file !== directory && !file.startsWith(`${directory}${path.sep}`)) { response.writeHead(403).end('Forbidden'); return; }
  if (existsSync(file) && statSync(file).isDirectory()) {
    if (!pathname.endsWith('/')) { response.writeHead(301, { Location: `${pathname}/` }).end(); return; }
    file = path.join(file, 'index.html');
  }
  if (!existsSync(file) || !statSync(file).isFile()) { response.writeHead(404).end('Not found'); return; }
  response.writeHead(200, { 'Content-Type': types[path.extname(file).toLowerCase()] || 'application/octet-stream' });
  createReadStream(file).pipe(response);
}).listen(port, '127.0.0.1', () => console.log(`Serving dist on port ${port}`));
