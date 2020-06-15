export function save(key, object) {
    localStorage.setItem(key, object);
    return object;
}
export function get(key) {
    return localStorage.getItem(key);
}

export function remove(key) {
    localStorage.removeItem(key);
}
