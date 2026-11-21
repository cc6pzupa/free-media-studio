// Injected by the Android app into the local studio pages. WebView ignores
// <a download>, so downloads are read here and streamed to CCAndroid, which
// writes them to Downloads/CC Media Studio.
(function () {
  if (window.__ccAndroidBridge || !window.CCAndroid) return;
  window.__ccAndroidBridge = true;
  // The pages revoke object URLs right after click(); keep the Blob so the
  // asynchronous save below can still read it.
  const blobs = new Map();
  const create = URL.createObjectURL.bind(URL);
  const revoke = URL.revokeObjectURL.bind(URL);
  URL.createObjectURL = function (object) {
    const url = create(object);
    if (object instanceof Blob) blobs.set(url, object);
    return url;
  };
  URL.revokeObjectURL = function (url) {
    setTimeout(function () { blobs.delete(url); revoke(url); }, 120000);
  };
  async function source(href) {
    if (blobs.has(href)) return blobs.get(href);
    const response = await fetch(href, { credentials: 'same-origin' });
    if (!response.ok) throw new Error('HTTP ' + response.status);
    return response.blob();
  }
  // A multiple of 3 bytes, so every chunk encodes to standalone base64.
  const STEP = 768 * 1024;
  async function save(href, name) {
    let id = -1;
    try {
      const blob = await source(href);
      id = CCAndroid.begin(name || 'download', blob.type || 'application/octet-stream');
      if (id < 0) throw new Error('storage unavailable');
      for (let offset = 0; offset < blob.size; offset += STEP) {
        const bytes = new Uint8Array(await blob.slice(offset, offset + STEP).arrayBuffer());
        let text = '';
        for (let i = 0; i < bytes.length; i += 32768) text += String.fromCharCode.apply(null, bytes.subarray(i, i + 32768));
        if (!CCAndroid.chunk(id, btoa(text))) throw new Error('write failed');
      }
      if (!CCAndroid.finish(id)) throw new Error('write failed');
    } catch (error) {
      if (id >= 0) CCAndroid.abort(id);
      CCAndroid.failed(String((error && error.message) || error));
    }
  }
  function handle(anchor) {
    if (!anchor || !anchor.hasAttribute || !anchor.hasAttribute('download') || !anchor.href) return false;
    save(anchor.href, anchor.getAttribute('download') || 'download');
    return true;
  }
  const click = HTMLAnchorElement.prototype.click;
  HTMLAnchorElement.prototype.click = function () { if (!handle(this)) click.call(this); };
  document.addEventListener('click', function (event) {
    const anchor = event.target && event.target.closest ? event.target.closest('a[download]') : null;
    if (anchor && handle(anchor)) { event.preventDefault(); event.stopPropagation(); }
  }, true);
})();
