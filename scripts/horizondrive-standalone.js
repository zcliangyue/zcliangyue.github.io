/* global hexo */
'use strict';

const fs = require('fs');
const path = require('path');

const srcPath = () => path.join(hexo.source_dir, 'HorizonDrive', 'index.html');
const dstPath = () => path.join(hexo.public_dir, 'HorizonDrive', 'index.html');

function copyStandalone({ log } = { log: true }) {
  const src = srcPath();
  const dst = dstPath();
  if (!fs.existsSync(src)) return;
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  fs.copyFileSync(src, dst);
  if (log) hexo.log.info('HorizonDrive: restored standalone public/HorizonDrive/index.html');
}

/**
 * index.md（Fluid layout）会覆盖与独立 index.html 相同的输出路径。
 * - after_generate（靠后执行）：hexo server 每次 watch 生成后写回独立页
 * - before_exit：hexo generate 结束前再覆盖一次（无日志，避免重复）
 */
hexo.extend.filter.register('after_generate', () => copyStandalone({ log: true }), 10000);
hexo.extend.filter.register('before_exit', () => copyStandalone({ log: false }));
